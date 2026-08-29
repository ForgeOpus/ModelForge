/* Mirrors the FastAPI contract in web/app.py + workflow/models.py + recommendation_engine. */

export type TrainingObjective = "supervised" | "preference" | "reinforcement";
export type ParallelismStrategy = "data_parallel" | "fsdp";
export type PeftApproach =
  | "full_finetune"
  | "lora_bf16"
  | "qlora_8bit"
  | "qlora_4bit";
export type RunStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface HealthResponse {
  status: string;
  version: string;
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
  hub_task_filter: string | null;
  compatible_objectives: TrainingObjective[];
  data_formats: Record<string, DataFormatSpec>;
}

export interface StrategyInfo {
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
  strategies: StrategyInfo[];
  use_cases: UseCase[];
}

export interface Gpu {
  name: string;
  total_memory_gb: number | "N/A";
}

export interface HardwareSpecs {
  physical_core_count: number;
  disk_storage_gb: number;
  available_disk_storage_gb: number;
  ram_gb: number;
  gpus: Gpu[];
}

export interface RecommendationRequest {
  use_case_slug: string;
  objective?: TrainingObjective;
  gpu_indices?: number[] | null;
  n?: number;
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
  /** May be an empty string when parallelism is trivial. */
  parallelism_decision: string;
  model_filter: string;
  warnings: string[];
}

export interface Recommendation {
  recommended_models: RecommendedModel[];
  task_slug: string;
  strategy_slug: string;
  peft_approach: PeftApproach | string;
  provider_slug: string;
  strategy_config: Record<string, unknown>;
  peft_config: Record<string, unknown> | null;
  data_format: DataFormatSpec;
  parallelism_strategy: ParallelismStrategy | string;
  rationale: Rationale;
}

export interface DatasetValidationRequest {
  path: string;
  remote?: boolean;
  use_case_slug?: string;
  objective?: TrainingObjective;
}

export interface DatasetValidationResult {
  valid: boolean;
  path: string;
  remote: boolean;
  columns: string[];
  format_description: string | null;
  error: string | null;
}

export interface TrainingRunRequest {
  use_case_slug: string;
  objective?: TrainingObjective;
  model_id: string;
  dataset_path: string;
  dataset_remote?: boolean;
  output_dir: string;
  gpu_indices?: number[];
  tensorboard?: boolean;
  tensorboard_log_dir?: string | null;
  recommendation?: Recommendation | Record<string, unknown> | null;
  resume_from_checkpoint?: string | null;
  parallelism_strategy_override?: ParallelismStrategy | null;
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
  /** JSON string of the Recommendation used to launch the run. */
  config_json: string;
  tensorboard_log_dir: string | null;
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

/** Safely parse RunRecord.config_json into a Recommendation. */
export function parseRunConfig(raw: string | null | undefined): Recommendation | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && "rationale" in parsed) {
      return parsed as Recommendation;
    }
    return null;
  } catch {
    return null;
  }
}
