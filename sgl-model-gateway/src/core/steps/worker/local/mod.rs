mod create_worker;
mod detect_connection;
mod discover_dp;
mod discover_metadata;
mod find_worker_to_update;
mod find_workers_to_remove;
mod remove_from_policy_registry;
mod remove_from_worker_registry;
mod submit_tokenizer_job;
mod update_policies_for_worker;
mod update_remaining_policies;
mod update_worker_properties;

use std::{sync::Arc, time::Duration};

/// Strip protocol prefix (http://, https://, grpc://) from URL.
pub(crate) fn strip_protocol(url: &str) -> String {
    url.trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_start_matches("grpc://")
        .to_string()
}

pub use create_worker::CreateLocalWorkerStep;
pub use detect_connection::DetectConnectionModeStep;
pub use discover_dp::{get_dp_info, DiscoverDPInfoStep, DpInfo};
pub use discover_metadata::DiscoverMetadataStep;
pub use find_worker_to_update::FindWorkerToUpdateStep;
pub use find_workers_to_remove::{FindWorkersToRemoveStep, WorkerRemovalRequest};
pub use remove_from_policy_registry::RemoveFromPolicyRegistryStep;
pub use remove_from_worker_registry::RemoveFromWorkerRegistryStep;
pub use submit_tokenizer_job::SubmitTokenizerJobStep;
pub use update_policies_for_worker::UpdatePoliciesForWorkerStep;
pub use update_remaining_policies::UpdateRemainingPoliciesStep;
pub use update_worker_properties::UpdateWorkerPropertiesStep;
use wfaas::{BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, WorkflowDefinition};

use super::shared::{ActivateWorkersStep, RegisterWorkersStep, UpdatePoliciesStep};
use crate::{
    app_context::AppContext,
    config::RouterConfig,
    core::{
        steps::workflow_data::{
            LocalWorkerWorkflowData, WorkerRemovalWorkflowData, WorkerUpdateWorkflowData,
        },
        Worker, WorkerRegistry,
    },
    protocols::worker_spec::{WorkerConfigRequest, WorkerUpdateRequest},
};

/// Find workers by URL, supporting both DP-aware (prefix match) and regular (exact match) modes.
///
/// For DP-aware workers, finds all workers with URL prefix `{url}@`.
/// For regular workers, finds the single worker with exact URL match.
pub(crate) fn find_workers_by_url(
    registry: &WorkerRegistry,
    url: &str,
    dp_aware: bool,
) -> Vec<Arc<dyn Worker>> {
    if dp_aware {
        let worker_url_prefix = format!("{}@", url);
        registry
            .get_all()
            .iter()
            .filter(|worker| worker.url().starts_with(&worker_url_prefix))
            .cloned()
            .collect()
    } else {
        match registry.get_by_url(url) {
            Some(worker) => vec![worker],
            None => Vec::new(),
        }
    }
}

pub fn create_local_worker_workflow(
    router_config: &RouterConfig,
) -> WorkflowDefinition<LocalWorkerWorkflowData> {
    let detect_timeout = Duration::from_secs(router_config.worker_startup_timeout_secs);

    // Calculate max_attempts based on timeout
    let timeout_secs = detect_timeout.as_secs() as f64;
    let effective_timeout = timeout_secs * 0.9;
    let max_attempts = if effective_timeout > 10.0 {
        (5 + ((effective_timeout - 10.0) / 5.0).ceil() as u32).max(3)
    } else {
        3
    };

    WorkflowDefinition::new("local_worker_registration", "Local Worker Registration")
        // Step 1: Detect connection mode (HTTP vs gRPC)
        .add_step(
            StepDefinition::new(
                "detect_connection_mode",
                "Detect Connection Mode",
                Arc::new(DetectConnectionModeStep),
            )
            .with_retry(RetryPolicy {
                max_attempts,
                backoff: BackoffStrategy::Linear {
                    increment: Duration::from_secs(1),
                    max: Duration::from_secs(5),
                },
            })
            .with_timeout(detect_timeout)
            .with_failure_action(FailureAction::FailWorkflow),
        )
        // Step 2a: Discover metadata (parallel with DP discovery)
        .add_step(
            StepDefinition::new(
                "discover_metadata",
                "Discover Metadata",
                Arc::new(DiscoverMetadataStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["detect_connection_mode"]),
        )
        // Step 2b: Discover DP info (after metadata to avoid concurrent /server_info calls)
        .add_step(
            StepDefinition::new(
                "discover_dp_info",
                "Discover DP Info",
                Arc::new(DiscoverDPInfoStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["discover_metadata"]),
        )
        // Step 3: Create worker(s)
        .add_step(
            StepDefinition::new(
                "create_worker",
                "Create Worker",
                Arc::new(CreateLocalWorkerStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["discover_dp_info"]),
        )
        // Step 4: Register workers (shared step)
        .add_step(
            StepDefinition::new(
                "register_workers",
                "Register Workers",
                Arc::new(RegisterWorkersStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["create_worker"]),
        )
        .add_step(
            StepDefinition::new(
                "submit_tokenizer_job",
                "Submit Tokenizer Job",
                Arc::new(SubmitTokenizerJobStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["register_workers"]),
        )
        // Step 5a: Update policies (parallel with activation)
        .add_step(
            StepDefinition::new(
                "update_policies",
                "Update Policies",
                Arc::new(UpdatePoliciesStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["register_workers"]),
        )
        // Step 5b: Activate workers (parallel with policy update)
        .add_step(
            StepDefinition::new(
                "activate_workers",
                "Activate Workers",
                Arc::new(ActivateWorkersStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["register_workers"]),
        )
}

/// Create a worker removal workflow definition.
///
/// DAG structure:
/// ```text
///     find_workers_to_remove
///              │
///     remove_from_policy_registry
///              │
///     remove_from_worker_registry
///              │
///     update_remaining_policies
/// ```
pub fn create_worker_removal_workflow() -> WorkflowDefinition<WorkerRemovalWorkflowData> {
    WorkflowDefinition::new("worker_removal", "Remove worker from router")
        .add_step(
            StepDefinition::new(
                "find_workers_to_remove",
                "Find workers to remove",
                Arc::new(FindWorkersToRemoveStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            }),
        )
        .add_step(
            StepDefinition::new(
                "remove_from_policy_registry",
                "Remove workers from policy registry",
                Arc::new(RemoveFromPolicyRegistryStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            })
            .depends_on(&["find_workers_to_remove"]),
        )
        .add_step(
            StepDefinition::new(
                "remove_from_worker_registry",
                "Remove workers from worker registry",
                Arc::new(RemoveFromWorkerRegistryStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            })
            .depends_on(&["remove_from_policy_registry"]),
        )
        .add_step(
            StepDefinition::new(
                "update_remaining_policies",
                "Update cache-aware policies for remaining workers",
                Arc::new(UpdateRemainingPoliciesStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            })
            .depends_on(&["remove_from_worker_registry"]),
        )
}

/// Create a worker update workflow definition.
///
/// DAG structure:
/// ```text
///     find_worker_to_update
///              │
///     update_worker_properties
///              │
///     update_policies_for_worker
/// ```
pub fn create_worker_update_workflow() -> WorkflowDefinition<WorkerUpdateWorkflowData> {
    WorkflowDefinition::new("worker_update", "Update worker properties")
        .add_step(
            StepDefinition::new(
                "find_worker_to_update",
                "Find worker to update",
                Arc::new(FindWorkerToUpdateStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            }),
        )
        .add_step(
            StepDefinition::new(
                "update_worker_properties",
                "Update worker properties",
                Arc::new(UpdateWorkerPropertiesStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            })
            .depends_on(&["find_worker_to_update"]),
        )
        .add_step(
            StepDefinition::new(
                "update_policies_for_worker",
                "Update policies for updated worker",
                Arc::new(UpdatePoliciesForWorkerStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            })
            .depends_on(&["update_worker_properties"]),
        )
}

/// Helper to create initial workflow data for local worker registration
pub fn create_local_worker_workflow_data(
    config: WorkerConfigRequest,
    app_context: Arc<AppContext>,
) -> LocalWorkerWorkflowData {
    LocalWorkerWorkflowData {
        config,
        connection_mode: None,
        discovered_labels: std::collections::HashMap::new(),
        dp_info: None,
        workers: None,
        final_labels: std::collections::HashMap::new(),
        detected_runtime_type: None,
        app_context: Some(app_context),
        actual_workers: None,
    }
}

/// Helper to create initial workflow data for worker removal
pub fn create_worker_removal_workflow_data(
    url: String,
    dp_aware: bool,
    app_context: Arc<AppContext>,
) -> WorkerRemovalWorkflowData {
    WorkerRemovalWorkflowData {
        config: WorkerRemovalRequest { url, dp_aware },
        workers_to_remove: None,
        worker_urls: Vec::new(),
        affected_models: std::collections::HashSet::new(),
        app_context: Some(app_context),
        actual_workers_to_remove: None,
    }
}

/// Helper to create initial workflow data for worker update
pub fn create_worker_update_workflow_data(
    worker_url: String,
    update_config: WorkerUpdateRequest,
    app_context: Arc<AppContext>,
) -> WorkerUpdateWorkflowData {
    // Determine if this is a DP-aware update based on URL pattern
    let dp_aware = worker_url.contains('@');
    WorkerUpdateWorkflowData {
        config: update_config,
        worker_url,
        dp_aware,
        app_context: Some(app_context),
        workers_to_update: None,
        updated_workers: None,
    }
}

#[cfg(test)]
mod tests {
    //! INF-425 regression coverage: an optional step's failure must not strand
    //! the steps that depend on it.
    //!
    //! `discover_metadata` is declared `FailureAction::ContinueNextStep` because
    //! a worker is worth registering even when its metadata cannot be read. The
    //! workflow engine has to honour that. wfaas 1.0.0 did not -- it signalled a
    //! `ContinueNextStep` failure to dependents as `StepResult::Failure`, and
    //! dependents are only woken on `Success | Skip`, so `create_worker` was
    //! never scheduled and the whole registration ended deadlock-detected. On
    //! the 2026-08-31 solar-pro4-prod decode roll that lost 13 of 26 new
    //! decoders, each one permanently: service discovery keeps a pod whose add
    //! failed in its tracked set, so every later watch event is dedupped away.
    //!
    //! INF-432 narrowed what "worth registering" means without weakening that.
    //! Reaching `create_worker` is still mandatory -- that is the scheduling
    //! property above, and the first test here. What `create_worker` then does
    //! is a separate decision: in IGW mode routing selects by model name, so a
    //! worker registered under `UNKNOWN_MODEL_ID` can never be picked and no
    //! later pass corrects it, which cost solar-pro4-prod ~40% of its fleet on
    //! 2026-09-01. It now refuses instead, leaving the pod unregistered -- the
    //! one state the service-discovery resync retries. Outside IGW the model
    //! filter is off and an unnamed worker still serves, so the fallback stays.

    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
        time::{Duration, Instant},
    };

    use async_trait::async_trait;
    use wfaas::{
        InMemoryStore, StepExecutor, StepId, StepResult, WorkflowContext, WorkflowEngine,
        WorkflowError, WorkflowId, WorkflowInstanceId, WorkflowResult, WorkflowState,
        WorkflowStatus,
    };

    use crate::core::ConnectionMode;

    use super::*;

    type Engine = WorkflowEngine<LocalWorkerWorkflowData, InMemoryStore<LocalWorkerWorkflowData>>;

    /// Which steps ran, in order. The regression is a step that never runs, not
    /// a step that fails, so the final status alone would not catch it.
    #[derive(Default)]
    struct ExecutionLog(Mutex<Vec<String>>);

    impl ExecutionLog {
        fn record(&self, step_id: &str) {
            self.0.lock().unwrap().push(step_id.to_string());
        }

        fn ran(&self, step_id: &str) -> bool {
            self.0.lock().unwrap().iter().any(|s| s == step_id)
        }

        fn order(&self) -> Vec<String> {
            self.0.lock().unwrap().clone()
        }
    }

    /// Stands in for one real step: records that it ran, then succeeds or fails
    /// on command. Keeps the workflow off the network without changing its shape.
    struct ScriptedStep {
        step_id: String,
        log: Arc<ExecutionLog>,
        fails: bool,
    }

    #[async_trait]
    impl StepExecutor<LocalWorkerWorkflowData> for ScriptedStep {
        async fn execute(
            &self,
            _context: &mut WorkflowContext<LocalWorkerWorkflowData>,
        ) -> WorkflowResult<StepResult> {
            self.log.record(&self.step_id);
            if self.fails {
                return Err(WorkflowError::StepFailed {
                    step_id: StepId::new(self.step_id.clone()),
                    message: format!("scripted failure in {}", self.step_id),
                });
            }
            Ok(StepResult::Success)
        }

        fn is_retryable(&self, _error: &WorkflowError) -> bool {
            true
        }
    }

    /// The production definition with only its executors replaced, so the DAG
    /// edges, retry policies, timeouts and failure actions under test are the
    /// ones the router actually runs.
    fn scripted_workflow(
        failing_step: Option<&str>,
        log: &Arc<ExecutionLog>,
    ) -> WorkflowDefinition<LocalWorkerWorkflowData> {
        let mut definition = create_local_worker_workflow(&RouterConfig::default());
        if let Some(failing_step) = failing_step {
            assert!(
                definition
                    .steps
                    .iter()
                    .any(|step| step.id.to_string() == failing_step),
                "{failing_step} is not a step of local_worker_registration"
            );
        }
        for step in &mut definition.steps {
            let step_id = step.id.to_string();
            step.executor = Arc::new(ScriptedStep {
                fails: failing_step == Some(step_id.as_str()),
                step_id,
                log: Arc::clone(log),
            });
        }
        definition
    }

    fn worker_data() -> LocalWorkerWorkflowData {
        LocalWorkerWorkflowData {
            config: serde_json::from_value(serde_json::json!({
                "url": "http://scripted-worker:30000"
            }))
            .expect("a url-only worker config should deserialize"),
            connection_mode: None,
            discovered_labels: HashMap::new(),
            dp_info: None,
            workers: None,
            final_labels: HashMap::new(),
            detected_runtime_type: None,
            // No scripted step reads the app context.
            app_context: None,
            actual_workers: None,
        }
    }

    /// Run the workflow to a terminal status and hand back what ran.
    /// `WorkflowEngine::wait_for_completion` drops the terminal state, which
    /// would leave nothing to assert on, so poll `get_status` instead.
    async fn run_local_worker_workflow(
        failing_step: Option<&str>,
    ) -> (Arc<ExecutionLog>, WorkflowState<LocalWorkerWorkflowData>) {
        let log = Arc::new(ExecutionLog::default());
        let engine = Engine::new();
        engine
            .register_workflow(scripted_workflow(failing_step, &log))
            .expect("local_worker_registration should validate");

        let instance = engine
            .start_workflow(WorkflowId::new("local_worker_registration"), worker_data())
            .await
            .expect("workflow should start");

        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let state = engine
                .get_status(instance)
                .await
                .expect("workflow state should be readable");
            if !matches!(
                state.status,
                WorkflowStatus::Pending | WorkflowStatus::Running
            ) {
                return (log, state);
            }
            assert!(
                Instant::now() < deadline,
                "workflow never reached a terminal status; ran {:?}",
                log.order()
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// The regression itself: the scheduling property, with every executor
    /// scripted, so what is under test is the DAG and not any step's own logic.
    #[tokio::test]
    async fn optional_metadata_failure_still_schedules_create_worker() {
        let (log, state) = run_local_worker_workflow(Some("discover_metadata")).await;

        assert!(
            log.ran("discover_metadata"),
            "the scripted step never ran, so the test proves nothing; ran {:?}",
            log.order()
        );
        assert!(
            log.ran("create_worker"),
            "discover_metadata is ContinueNextStep, so its failure must not stop \
             create_worker (INF-425); ran {:?}",
            log.order()
        );
        assert!(
            log.ran("register_workers"),
            "the worker must reach the registry; ran {:?}",
            log.order()
        );
        assert!(
            log.ran("activate_workers"),
            "the worker must be activated to take traffic; ran {:?}",
            log.order()
        );
        assert_eq!(
            state.status,
            WorkflowStatus::Completed,
            "the workflow must not deadlock when an optional step fails; ran {:?}",
            log.order()
        );
    }

    /// Build an AppContext that differs from the default only in IGW mode.
    async fn app_context(enable_igw: bool) -> Arc<AppContext> {
        let config = RouterConfig {
            enable_igw,
            ..RouterConfig::default()
        };
        Arc::new(
            AppContext::from_config(config, 60)
                .await
                .expect("a default AppContext should build"),
        )
    }

    /// The real `create_worker`, over data that carries no model identity --
    /// what `discover_metadata` leaves behind when a booting worker answers
    /// neither `/model_info` nor `/server_info`. No network is involved.
    async fn create_worker_without_identity(enable_igw: bool) -> WorkflowResult<StepResult> {
        let mut data = worker_data();
        data.connection_mode = Some(ConnectionMode::Http);
        data.app_context = Some(app_context(enable_igw).await);
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        CreateLocalWorkerStep.execute(&mut context).await
    }

    /// INF-432: registering it would hide a lost worker behind a healthy entry.
    #[tokio::test]
    async fn igw_refuses_a_worker_with_no_model_identity() {
        let error = create_worker_without_identity(true)
            .await
            .expect_err("IGW cannot route to an unnamed worker, so it must not register one");

        let message = error.to_string();
        assert!(
            message.contains("no model identity"),
            "the failure has to name its cause, or the operator sees only a failed \
             AddWorker; got {message}"
        );
    }

    /// The other side of the same decision, so the narrowing stays narrow.
    #[tokio::test]
    async fn without_igw_an_unnamed_worker_still_registers() {
        create_worker_without_identity(false)
            .await
            .expect("outside IGW the model filter is off, so an unnamed worker still serves");
    }

    #[tokio::test]
    async fn every_step_runs_when_nothing_fails() {
        let (log, state) = run_local_worker_workflow(None).await;

        assert_eq!(state.status, WorkflowStatus::Completed);
        for step_id in [
            "detect_connection_mode",
            "discover_metadata",
            "discover_dp_info",
            "create_worker",
            "register_workers",
            "submit_tokenizer_job",
            "update_policies",
            "activate_workers",
        ] {
            assert!(
                log.ran(step_id),
                "{step_id} did not run; ran {:?}",
                log.order()
            );
        }
    }

    /// The other direction: the harness has to be able to see a workflow fail,
    /// or the test above would pass for the wrong reason.
    #[tokio::test]
    async fn a_fail_workflow_step_does_stop_the_workflow() {
        let (log, state) = run_local_worker_workflow(Some("create_worker")).await;

        assert_eq!(
            state.status,
            WorkflowStatus::Failed,
            "create_worker is FailWorkflow; ran {:?}",
            log.order()
        );
        assert!(
            !log.ran("register_workers"),
            "nothing downstream of a FailWorkflow step should run; ran {:?}",
            log.order()
        );
    }
}
