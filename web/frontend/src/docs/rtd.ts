// Each in-app page has a longer counterpart on Read the Docs (docs/ at the repo root).
// Keep this map and the "Keeping the two doc sets in sync" table in
// docs/contributing/contributing.md up to date when adding or renaming pages.
export const RTD_BASE='https://modelforge-finetuning.readthedocs.io/en/latest/';

export const RTD_PAGES:Record<string,string>={
  'introduction':'',
  'install-and-first-run':'getting-started/quickstart/',
  'tour-of-the-console':'user-guide/browser-console/',
  'using-the-playground':'user-guide/playground/',
  'what-fine-tuning-is':'getting-started/concepts/',
  'objectives':'strategies/overview/',
  'use-cases-and-tasks':'configuration/training-tasks/',
  'preparing-a-dataset':'configuration/dataset-formats/',
  'where-vram-goes':'configuration/hardware-profiles/',
  'full-fine-tuning':'strategies/full-finetuning/',
  'lora-and-qlora':'strategies/qlora/',
  'how-a-tier-is-chosen':'configuration/configuration-guide/',
  'picking-a-base-model':'configuration/configuration-guide/',
  'data-parallel-vs-fsdp':'strategies/multi-gpu/',
  'gradient-checkpointing-and-packing':'strategies/memory-optimizations/',
  'liger-kernels':'strategies/memory-optimizations/',
  'resuming-from-checkpoints':'user-guide/checkpoints/',
  'fixing-out-of-memory-errors':'troubleshooting/performance/',
  'recommendation-fields':'api-reference/responses/',
  'training-config-keys':'api-reference/training-config/',
  'cli-commands':'api-reference/cli/',
  'http-api':'api-reference/rest-api/',
};

export const rtdUrl=(slug:string)=>RTD_BASE+(RTD_PAGES[slug]??'');
