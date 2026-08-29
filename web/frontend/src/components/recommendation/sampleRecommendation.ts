import type { Recommendation } from "../../types/api";

export const sampleRecommendation: Recommendation = {
  task_slug: "causal_language_modeling",
  strategy_slug: "sft",
  peft_approach: "qlora_4bit",
  provider_slug: "huggingface",
  parallelism_strategy: "data_parallel",
  recommended_models: [
    { model_id: "mistralai/Mistral-7B-v0.3", num_parameters: 7_248_023_552, downloads: 1_284_113 },
    { model_id: "meta-llama/Llama-3.2-3B", num_parameters: 3_212_749_824, downloads: 842_665 },
    { model_id: "Qwen/Qwen2.5-7B", num_parameters: 7_615_616_512, downloads: 693_204 },
    { model_id: "google/gemma-2-2b", num_parameters: 2_614_341_888, downloads: 421_889 },
  ],
  strategy_config: {
    learning_rate: 0.0002,
    num_train_epochs: 3,
    per_device_train_batch_size: 2,
    gradient_accumulation_steps: 8,
    max_seq_length: 2048,
    gradient_checkpointing: true,
  },
  peft_config: {
    r: 16,
    lora_alpha: 32,
    lora_dropout: 0.05,
    bias: "none",
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"],
  },
  data_format: {
    user_facing_description:
      "A JSONL file where each row is a chat with a 'prompt' and a 'response'. The prompt holds the user instruction; the response holds the target completion the model learns to produce.",
    column_aliases: {
      instruction: "prompt",
      input: "prompt",
      output: "response",
    },
  },
  rationale: {
    vram_summary:
      "Your NVIDIA RTX 4090 has 24.0 GB of VRAM. QLoRA (4-bit) training of a 7B model needs about 16.0 GB, which fits but leaves only 8.0 GB of headroom for activations and longer sequences.",
    use_case_resolution:
      "instruction_following → causal_language_modeling (supervised objective)",
    peft_decision:
      "With 24.0 GB of VRAM the parameter budget P_max works out to roughly 7B for a 4-bit adapter. full_finetune and lora_bf16 both exceed that budget for a 7B model, and qlora_8bit needs about 22 GB. qlora_4bit is the highest-fidelity tier that fits, so the engine selected the 4-bit path.",
    parallelism_decision: "",
    model_filter:
      "text-generation models under 8B parameters, sorted by downloads; the size ceiling was relaxed from 7B to 8B on the retry pass.",
    warnings: [
      "VRAM headroom is tight (8.0 GB). Reduce max_seq_length or per-device batch size if you hit out-of-memory errors.",
    ],
  },
};
