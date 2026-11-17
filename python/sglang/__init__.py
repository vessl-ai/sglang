"""SGLang public API."""

import platform as _platform
import sys as _sys

import os as _os

if _os.environ.get("SGLANG_OPT_INF834_EARLY_CUINIT", "").lower() in ("1", "true"):
    # INF-834: start the CUDA driver init (cuInit, seconds on an 8-GPU node) on a
    # thread now, so it overlaps this process's Python imports instead of
    # running inside the first torch.cuda.is_available() call (a module-level
    # is_cuda() reached during the imports). cuInit is process-wide and
    # idempotent; it creates no context and allocates no device memory.
    from sglang._inf834_early_cuinit import start as _inf834_start_cuinit

    _inf834_start_cuinit()

if _os.environ.get("SGLANG_OPT_INF834_STANDBY_CHILDREN", "0") not in ("", "0"):
    # INF-834: standby scheduler / DP controller children that import while this
    # (main) process imports; see sglang/_inf834_standby.py.
    from sglang._inf834_standby import maybe_start as _inf834_standby_start

    _inf834_standby_start()

# sglang.srt.environ must run before the rest of this file's imports
# (hf_transformers_patches, lang.api, ...), which pull in torch and
# FlashInfer: those claim these cache dirs early, and the first value set is
# the one that sticks. Safe here -- environ has no heavy dependency (no torch).
from sglang.srt.environ import (
    redirect_third_party_caches as _redirect_third_party_caches,
)

_redirect_third_party_caches()

if _sys.platform == "darwin" and _platform.machine() == "arm64":
    from sglang._platform_stubs import install_platform_stubs as _install_platform_stubs

    _install_platform_stubs()

from sglang.srt.utils.hf_transformers_patches import apply_all as _apply_hf_patches

_apply_hf_patches()

from sglang.lang.api import (
    Runtime,
    assistant,
    assistant_begin,
    assistant_end,
    flush_cache,
    function,
    gen,
    gen_int,
    gen_string,
    get_server_info,
    image,
    select,
    separate_reasoning,
    set_default_backend,
    system,
    system_begin,
    system_end,
    user,
    user_begin,
    user_end,
    video,
)
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.lang.choices import (
    greedy_token_selection,
    token_length_normalized,
    unconditional_likelihood_normalized,
)
from sglang.lang.global_config import global_config

# Lazy backend clients
from sglang.utils import LazyImport
from sglang.version import __version__

Anthropic = LazyImport("sglang.lang.backend.anthropic", "Anthropic")
Crusoe = LazyImport("sglang.lang.backend.crusoe", "Crusoe")
LiteLLM = LazyImport("sglang.lang.backend.litellm", "LiteLLM")
OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")
VertexAI = LazyImport("sglang.lang.backend.vertexai", "VertexAI")

# Runtime API
ServerArgs = LazyImport("sglang.srt.server_args", "ServerArgs")
Engine = LazyImport("sglang.srt.entrypoints.engine", "Engine")

__all__ = [
    "Anthropic",
    "Crusoe",
    "Engine",
    "LiteLLM",
    "OpenAI",
    "Runtime",
    "RuntimeEndpoint",
    "ServerArgs",
    "VertexAI",
    "__version__",
    "assistant",
    "assistant_begin",
    "assistant_end",
    "flush_cache",
    "function",
    "gen",
    "gen_int",
    "gen_string",
    "get_server_info",
    "global_config",
    "greedy_token_selection",
    "image",
    "select",
    "separate_reasoning",
    "set_default_backend",
    "system",
    "system_begin",
    "system_end",
    "token_length_normalized",
    "unconditional_likelihood_normalized",
    "user",
    "user_begin",
    "user_end",
    "video",
]
