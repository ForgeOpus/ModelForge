/* Mirrors workflow/models.py and recommendation_engine/validation_models/recommendation.py.
   Keep in sync when those change. */

export type RunStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type PeftApproach = "full_finetune" | "lora_bf16" | "qlora_8bit" | "qlora_4bit";

export interface Gpu {
  name: string;
  /** The detector returns the string "N/A" for Apple Silicon, which has no
      queryable VRAM figure. Every consumer must guard for it. */
  total_memory_gb: number | string;
}

export interface HardwareSpecs {
  physical_core_count: number;
  disk_storage_gb: number;
  available_disk_storage_gb: number;
  ram_gb: number;
  gpus: Gpu[];
}

export interface DataFormatSpec {
  user_facing_description: string;
  column_aliases: Record<string, string>;
}

export interface UseCase {
  name: string;
  slug: string;
  description: string;
  task_slug: string;
  hub_task_filter: string;
  compatible_objectives: string[];
  data_formats: Record<string, DataFormatSpec>;
}

export interface Strategy {
  name: string;
  slug: string;
  description: string;
  supported_providers: string[];
  supported_tasks: string[];
  extra_model_copies: number;
  batch_multiplier: number;
}

export interface SystemInfo {
  name: string;
  package: string;
  version: string;
  providers: string[];
  strategies: Strategy[];
  use_cases: UseCase[];
}

export interface RecommendedModel {
  model_id: string;
  num_parameters: number;
  downloads: number;
}

export interface Rationale {
  vram_summary: string;
  use_case_resolution: string;
  peft_decision: string;
  model_filter: string;
  warnings: string[];
}

export interface VramEstimate {
  raw_vram_gb: number;
  adjusted_vram_gb: number;
  activation_budget_gb: number;
  adapter_cost_gb: number;
  bytes_per_param: number;
  max_params_by_tier: Record<string, number>;
}

export interface Recommendation {
  recommended_models: RecommendedModel[];
  task_slug: string;
  strategy_slug: string;
  peft_approach: PeftApproach;
  provider_slug: string;
  strategy_config: Record<string, unknown>;
  peft_config: { r: number; lora_alpha: number; target_modules: string[] } | null;
  data_format: DataFormatSpec;
  rationale: Rationale;
  vram_estimate?: VramEstimate;
}

export interface RunRecord {
  id: number;
  status: RunStatus;
  model_id: string;
  use_case: string;
  objective: string;
  task_slug: string | null;
  strategy: string | null;
  peft: string | null;
  dataset: string;
  output_dir: string;
  config_json: string;
  tensorboard_log_dir: string | null;
  tensorboard_url?: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface CheckpointRecord {
  id: number;
  run_id: number;
  path: string;
  step: number | null;
  created_at: string;
}

export interface TrainingLogs {
  loss?: number;
  grad_norm?: number;
  learning_rate?: number;
  epoch?: number;
  num_tokens?: number;
  mean_token_accuracy?: number;
  train_runtime?: number;
  train_samples_per_second?: number;
  [key: string]: number | undefined;
}

export interface TrainingEvent {
  run_id: number;
  type: "status" | "log" | "checkpoint" | "cancel" | "completed" | "cancelled" | "error" | "heartbeat";
  message: string;
  payload: {
    step?: number;
    max_steps?: number;
    logs?: TrainingLogs;
    path?: string;
    output_dir?: string;
    error?: string;
  };
  /**
   * Unix timestamp in seconds, not milliseconds.
   * The backend sets this from `time.time()`. When it is absent, the client
   * fills `Date.now() / 1000` once at the event boundary before any ETA math.
   */
  ts?: number;
}

export interface DatasetValidation {
  valid: boolean;
  path: string;
  remote: boolean;
  columns: string[];
  row_count?: number | null;
  format_description: string | null;
  error: string | null;
}

export interface SavedModel {
  id: number;
  name: string;
  task_slug: string;
  description: string | null;
  model_path: string;
  pushed_to_hub: boolean;
  created_at: string;
  base_model?: string | null;
  peft_approach?: PeftApproach | null;
  size_bytes?: number | null;
  run_id?: number | null;
}
