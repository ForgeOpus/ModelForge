import {C, Callout, Code, D, Table, type DocPage} from '../ui';

export const reference:DocPage[]=[
{
  slug:'recommendation-fields',title:'Recommendation fields',group:'Reference',minutes:4,
  summary:'Every field returned by POST /api/recommendations and shown in the plan.',
  lead:<>A recommendation is the plan ModelForge builds from your hardware, use case and objective. The console's Plan panel displays it, and <C>POST /api/runs</C> trains from it.</>,
  sections:[
    {id:'top',title:'Top-level fields',body:<Table head={['Field','Type','Meaning']} rows={[
      ['recommended_models','list','Up to n candidates: model_id, num_parameters, downloads. The first one is the default choice.'],
      ['task_slug','string','causal_language_modeling, seq2seq_modeling or question_answering.'],
      ['strategy_slug','string','sft, dpo, grpo or qa.'],
      ['peft_approach','string','full_finetune, lora_bf16, qlora_8bit or qlora_4bit.'],
      ['provider_slug','string','Always huggingface.'],
      ['parallelism_strategy','string','data_parallel or fsdp.'],
      ['strategy_config','object | null','Training arguments for the top model. See Training config keys. Null when no model fits.'],
      ['peft_config','object | null','r, lora_alpha, target_modules. Null for full fine-tuning.'],
      ['data_format','object','user_facing_description, column_aliases and optional_column_aliases for the objective.'],
      ['rationale','object','Human-readable reasoning, below.'],
      ['vram_estimate','object','The numbers behind the tier choice, below.'],
    ]}/>},
    {id:'rationale',title:'rationale',body:<Table head={['Field','Shown in the console as']} rows={[
      ['vram_summary','VRAM estimate'],
      ['peft_decision','PEFT decision'],
      ['warnings','The list under the plan'],
      ['use_case_resolution','Not shown. How the use case mapped to a task and strategy.'],
      ['parallelism_decision','Not shown. Why data parallel or FSDP.'],
      ['model_filter','Not shown. The Hub filter and size ceiling used.'],
    ]}/>},
    {id:'vram',title:'vram_estimate',body:<Table head={['Field','Meaning']} rows={[
      ['raw_vram_gb','Total VRAM across the GPUs used (Apple MPS counts as 8 GB).'],
      ['adjusted_vram_gb','After the strategy\'s extra copies and batch multiplier. See Where VRAM goes.'],
      ['activation_budget_gb','2.0'],
      ['adapter_cost_gb','0.1'],
      ['bytes_per_param','18, 2, 1 and 0.5 for the four tiers.'],
      ['max_params_by_tier','Parameter ceiling for every tier. The chosen tier\'s value filters the model search.'],
    ]}/>},
  ],
},
{
  slug:'training-config-keys',title:'Training config keys',group:'Reference',minutes:6,
  summary:'strategy_config keys and defaults for SFT, DPO, GRPO and QA.',
  lead:<>These are the values ModelForge puts in <C>strategy_config</C>. They're passed to the TRL or Transformers trainer as training arguments, and the Plan panel lists them under <b>Training config</b>.</>,
  sections:[
    {id:'common',title:'Shared by SFT, DPO and GRPO',body:<Table head={['Key','Default','Notes']} rows={[
      ['per_device_train_batch_size','1','Effective batch = this × gradient_accumulation_steps × GPUs.'],
      ['gradient_accumulation_steps','varies','See the table below.'],
      ['num_train_epochs','3.0',''],
      ['gradient_checkpointing','true','See Gradient checkpointing and packing.'],
      ['optim','adamw_torch',''],
      ['max_grad_norm','1.0','Gradient clipping.'],
      ['weight_decay','0.0',''],
      ['logging_steps','10','How often loss appears in the training log.'],
      ['save_steps','500','Checkpoint interval. All checkpoints are kept.'],
      ['seed','42',''],
      ['bf16 / fp16','by GPU','bf16 when every GPU has compute capability ≥ 8.0, otherwise fp16. Both off on CPU and MPS.'],
      ['use_liger_kernel','by install','SFT and DPO only. See Liger kernels.'],
    ]}/>},
    {id:'by-vram',title:'Scaled by total VRAM',body:<Table head={['Key','< 12 GB','12–32 GB','≥ 32 GB']} right={[1,2,3]} rows={[
      ['max_length (SFT, DPO)','512','1024','2048'],
      ['activation_offloading (SFT, DPO)','true','false','false'],
      ['gradient_accumulation_steps (SFT)','16','8','4'],
      ['gradient_accumulation_steps (DPO, GRPO)','32','16','8'],
      ['max_completion_length (GRPO)','128','256','512'],
    ]}/>},
    {id:'sft',title:'SFT',body:<Table head={['Key','Default']} rows={[
      ['learning_rate','2e-5'],
      ['lr_scheduler_type','linear'],
      ['warmup_ratio','0.0'],
      ['packing / packing_strategy','true / bfd (causal LMs only)'],
      ['model_init_kwargs','load_in_8bit or load_in_4bit for QLoRA tiers'],
    ]}/>},
    {id:'dpo',title:'DPO',body:<Table head={['Key','Default']} rows={[
      ['beta','0.1'],
      ['loss_type','sigmoid'],
      ['learning_rate','1e-7 full fine-tune. LoRA: 5e-6 below 1B · 5e-7 below 7B · 2e-7 below 13B · 1e-7 above.'],
      ['lr_scheduler_type / warmup_ratio','cosine / 0.1'],
    ]}/>},
    {id:'grpo',title:'GRPO',body:<Table head={['Key','Default']} rows={[
      ['num_generations','4'],
      ['beta','0.0 (no KL penalty)'],
      ['learning_rate','1e-6 full fine-tune, 5e-6 LoRA'],
      ['lr_scheduler_type / warmup_ratio','cosine / 0.1'],
      ['reward','Exact match after trimming whitespace'],
    ]}/>},
    {id:'qa',title:'QA',body:<Table head={['Key','≥ 12 GB','< 12 GB']} rows={[
      ['per_device_train_batch_size','4','1'],
      ['gradient_accumulation_steps','1','8'],
      ['gradient_checkpointing','false','true'],
      ['max_length','384','256'],
      ['doc_stride','128','128'],
      ['learning_rate','3e-5','3e-5'],
    ]}/>},
    {id:'editing',title:'Editing the config',body:<p>The console doesn't edit these values. To change them, send your own <C>recommendation</C> with <C>POST /api/runs</C>. See <D to="http-api">HTTP API</D>.</p>},
  ],
},
{
  slug:'cli-commands',title:'CLI commands',group:'Reference',minutes:3,
  summary:'modelforge run, cli, recommend, history, list-tasks and tensorboard.',
  lead:<>The <C>modelforge</C> command starts the browser console and offers a few utilities for the terminal. Running it with no arguments prints help.</>,
  sections:[
    {id:'run',title:'modelforge run',body:<>
      <p>Starts the server and browser console. See <D to="install-and-first-run">Install and first run</D>.</p>
      <Table head={['Flag','Default','Effect']} rows={[
        ['--host','0.0.0.0','Interface to bind.'],
        ['--port','8000','Port to bind.'],
        ['--no-browser','off','Don\'t open a browser.'],
        ['--token TOKEN','random','Fixed startup token.'],
      ]}/>
    </>},
    {id:'cli',title:'modelforge cli',body:<>
      <p>The interactive terminal wizard. It asks for a use case, objective, dataset, model and output folder, then trains in the terminal. It needs the <C>[cli]</C> extra and offers a Hugging Face login if none is cached.</p>
      <Table head={['Flag','Effect']} rows={[
        ['--no-tensorboard','Don\'t write TensorBoard logs.'],
        ['--tensorboard-logdir DIR','Where to write them.'],
      ]}/>
    </>},
    {id:'recommend',title:'modelforge recommend',body:<>
      <p>Prints a recommendation without training.</p>
      <Table head={['Flag','Default','Effect']} rows={[
        ['--use-case SLUG','prompt','instruction_following, text_generation, summarization or extractive_qa.'],
        ['--objective','supervised','supervised, preference or reinforcement.'],
        ['--n','5','Number of candidate models.'],
      ]}/>
      <Code>{`$ modelforge recommend --use-case instruction_following --objective preference`}</Code>
    </>},
    {id:'others',title:'Other commands',body:<Table head={['Command','Does']} rows={[
      ['modelforge list-tasks','Lists the available use cases.'],
      ['modelforge history','Prints browser run history. Add --legacy to see runs from the terminal wizard.'],
      ['modelforge tensorboard [LOGDIR]','Launches TensorBoard (default ./modelforge_output, --host 127.0.0.1, --port 6006).'],
      ['modelforge-nb','Notebook-friendly entry point.'],
    ]}/>},
  ],
},
{
  slug:'http-api',title:'HTTP API',group:'Reference',minutes:6,
  summary:'Authentication, REST endpoints and WebSocket events.',
  lead:<>Everything the console does goes through a JSON API on the same server. You can script it to queue runs from CI, or build your own front end.</>,
  sections:[
    {id:'auth',title:'Authentication',body:<>
      <p>Every endpoint except <C>/api/health</C> requires the startup token, sent in any one of these ways:</p>
      <Code>{`Authorization: Bearer <token>
X-ModelForge-Token: <token>
?token=<token>          # query string, required for WebSockets`}</Code>
      <p>A missing or wrong token returns <C>401</C>. Start the server with <C>--token</C> to keep the token fixed for scripts.</p>
    </>},
    {id:'example',title:'Example',body:<Code>{`$ curl -H "X-ModelForge-Token: $TOKEN" \\
    -H "Content-Type: application/json" \\
    -d '{"use_case_slug":"instruction_following","objective":"supervised"}' \\
    http://127.0.0.1:8000/api/recommendations`}</Code>},
    {id:'system',title:'System and planning',body:<Table head={['Endpoint','Purpose']} rows={[
      ['GET /api/health','Liveness and version. No auth.'],
      ['GET /api/system','Version, providers, strategies, use cases.'],
      ['GET /api/hardware','Detected GPUs, RAM and disk.'],
      ['GET /api/use-cases','Use cases with their objectives and data formats.'],
      ['POST /api/recommendations','{use_case_slug, objective, gpu_indices?, n=5} → a recommendation.'],
      ['POST /api/datasets/validate','{path, remote, use_case_slug?, objective} → {valid, columns, row_count, error}.'],
      ['POST /api/models/validate','{model_id, task_slug?} → {valid, num_parameters, pipeline_tag, error}.'],
    ]}/>},
    {id:'runs',title:'Runs',body:<>
      <Table head={['Endpoint','Purpose']} rows={[
        ['POST /api/runs','Start a run. 409 if one is already active.'],
        ['GET /api/runs?limit=100','List runs.'],
        ['GET /api/runs/{id}','One run.'],
        ['POST /api/runs/{id}/cancel','Stop after the current step.'],
        ['GET /api/runs/{id}/events','Recorded events.'],
        ['GET /api/runs/{id}/checkpoints','Saved checkpoints, newest first.'],
        ['POST /api/runs/{id}/resume','{checkpoint_path} → a new run continuing from it.'],
      ]}/>
      <p><C>POST /api/runs</C> takes <C>use_case_slug</C>, <C>objective</C>, <C>model_id</C>, <C>dataset_path</C>, <C>dataset_remote</C>, <C>output_dir</C>, <C>gpu_indices</C>, <C>tensorboard</C> (default true), <C>tensorboard_log_dir</C>, <C>recommendation</C>, <C>resume_from_checkpoint</C> and <C>parallelism_strategy_override</C>.</p>
    </>},
    {id:'ws',title:'Live events',body:<>
      <Code>{`WS /api/runs/{id}/events?token=<token>`}</Code>
      <p>Each message is <C>{'{run_id, type, message, payload, ts}'}</C>. Types: <C>status</C>, <C>log</C> (payload has <C>step</C>, <C>max_steps</C> and the trainer's <C>logs</C>), <C>checkpoint</C>, <C>cancel</C>, <C>completed</C>, <C>cancelled</C>, <C>error</C>, and a <C>heartbeat</C> every second while idle.</p>
    </>},
    {id:'models',title:'Models and Playground',body:<Table head={['Endpoint','Purpose']} rows={[
      ['GET /api/models, GET /api/models/{id}','Registered trained models.'],
      ['DELETE /api/models/{id}?delete_files=false','Remove from the registry, optionally deleting the folder.'],
      ['POST /api/models/{id}/push','{repo_id, private} → upload the output folder to the Hub.'],
      ['POST /api/playground/infer','{run_id, checkpoint_path?, context_window, max_new_tokens, messages} → {reply, input_tokens, context_limit, device}.'],
    ]}/>},
    {id:'notes',title:'Notes',body:<Callout tone="warn" title="Pushing uploads the whole folder.">That includes checkpoints and TensorBoard logs. Delete what you don't want to publish first.</Callout>},
  ],
},
];
