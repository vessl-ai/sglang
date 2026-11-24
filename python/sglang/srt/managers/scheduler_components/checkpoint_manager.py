"""Process checkpoint state of one scheduler.

An external orchestrator captures the whole process once it is prepared,
then restores it, in a process started from the capture or in place. The
scheduler goes running -> preparing -> prepared -> running any number of
times, or ends in failed. Repeated calls are no-ops.

The tokenizer pauses generation as /pause_generation does before preparing,
and continues it after restoring, so clients see a paused server in between.
Prepare waits for the paused scheduler to go idle, then releases the KV cache.
Weights stay on the GPU. Restore brings the KV cache back, then runs
`after_restore`, which builds what the checkpoint could not carry (the PD
transport, the HiCache L3 client) and raises if any registration failed.
"""

import logging
from typing import Any, Callable, List, Optional, Set

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.managers.io_struct import (
    CheckpointReqInput,
    CheckpointReqOutput,
    CheckpointReqType,
    CheckpointState,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)

logger = logging.getLogger(__name__)


class SchedulerCheckpointManager:
    def __init__(
        self,
        *,
        enable_memory_saver: bool,
        pp_size: int,
        offload_tags: Set[str],
        is_engine_paused: Callable[[], bool],
        is_fully_idle: Callable[..., bool],
        # This tree's release asserts the idle condition instead of returning
        # success=False, so a refused release raises and fails the prepare.
        release_memory_occupation: Callable[[ReleaseMemoryOccupationReqInput], None],
        resume_memory_occupation: Callable[[ResumeMemoryOccupationReqInput], None],
        all_gather_object: Callable[[Any], List[Any]],
        after_restore: Optional[Callable[[CheckpointReqInput], None]] = None,
        before_release: Optional[Callable[[], None]] = None,
        # Why `before_release` would refuse before anything is released, or
        # None: checked with the other refusals, before generation is paused.
        prepare_refusal: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self._enable_memory_saver = enable_memory_saver
        self._pp_size = pp_size
        # Live view of the memory regions released through the admin API.
        self._offload_tags = offload_tags
        self._is_engine_paused = is_engine_paused
        self._is_fully_idle = is_fully_idle
        self._release_memory_occupation = release_memory_occupation
        self._resume_memory_occupation = resume_memory_occupation
        # Over the TP ranks of this DP rank.
        self._all_gather_object = all_gather_object
        self._after_restore = after_restore
        self._before_release = before_release
        self._prepare_refusal = prepare_refusal
        self.state = CheckpointState.RUNNING
        self._error = ""
        # Tokenizer workers whose startup warmup is done (STARTUP_READY).
        self._startup_ready_workers: Set[int] = set()

    def handle(self, recv_req: CheckpointReqInput) -> CheckpointReqOutput:
        # STATE reports whether PREPARE would be refused. Repeats are no-ops.
        success, message = True, ""
        if recv_req.req_type == CheckpointReqType.STARTUP_READY:
            self._startup_ready_workers.add(int(recv_req.worker_id or 0))
        elif self.state == CheckpointState.FAILED:
            success = False
        elif recv_req.req_type == CheckpointReqType.RESTORE:
            if self.state == CheckpointState.PREPARING:
                success, message = False, "Cannot restore while preparing"
            elif self.state == CheckpointState.PREPARED:
                logger.info("Restoring from a process checkpoint")
                try:
                    self._resume_memory_occupation(
                        ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
                    )
                    if self._after_restore is not None:
                        self._after_restore(recv_req)
                except Exception as e:
                    self.state = CheckpointState.FAILED
                    self._error = f"Failed to restore from the checkpoint: {e}"
                    logger.exception(self._error)
                else:
                    self.state = CheckpointState.RUNNING
        elif self.state == CheckpointState.RUNNING:
            if not self._enable_memory_saver:
                success = False
                message = "Preparing for a checkpoint requires --enable-memory-saver"
            elif self._pp_size > 1:
                success = False
                message = (
                    "Preparing for a checkpoint does not support pipeline parallelism"
                )
            elif self._offload_tags:
                success = False
                message = f"Memory is already released: {sorted(self._offload_tags)}"
            elif self._is_engine_paused() and not self._is_drained():
                # A paused scheduler no longer advances its grammar, KV transfer
                # or HiCache queues, so a retract pause not idle now never is.
                success = False
                message = (
                    "Cannot drain a retract pause while requests wait on grammar, "
                    "KV transfer or HiCache work. Use mode=abort"
                )
            elif (
                self._prepare_refusal is not None
                and (refusal := self._prepare_refusal()) is not None
            ):
                success, message = False, refusal
            elif recv_req.req_type == CheckpointReqType.PREPARE:
                logger.info("Preparing for a process checkpoint")
                self.state = CheckpointState.PREPARING
                self.check_pending()
        out = CheckpointReqOutput(
            success=success,
            message=message or self._error,
            state=self.state,
            ready_workers=len(self._startup_ready_workers),
        )
        # Only the first TP rank replies to the tokenizer, so it reports them all.
        # Every TP rank handles this request in the same scheduler iteration.
        return CheckpointReqOutput.merge(self._all_gather_object(out))

    def check_pending(self) -> None:
        """Finish preparing once the drain is complete."""
        if self.state != CheckpointState.PREPARING or not self._is_drained():
            return
        try:
            if self._before_release is not None:
                self._before_release()
            self._release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            )
        except Exception as e:
            self.state = CheckpointState.FAILED
            self._error = f"Failed to prepare for the checkpoint: {e}"
            logger.exception(self._error)
        else:
            self.state = CheckpointState.PREPARED
            logger.info("Prepared for a process checkpoint")

    def _is_drained(self) -> bool:
        # Same idle condition as release_memory_occupation asserts.
        return self._is_fully_idle(ignore_waiting=self._is_engine_paused())
