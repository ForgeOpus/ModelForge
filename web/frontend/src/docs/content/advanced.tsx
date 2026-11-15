import {C, Callout, Code, D, Table, type DocPage} from '../ui';
import {PackingDiagram, ParallelDiagram} from '../diagrams';

export const advanced:DocPage[]=[
{
  slug:'data-parallel-vs-fsdp',title:'Data parallel vs. FSDP',group:'Advanced',minutes:8,readFirst:'where-vram-goes',
  summary:'Training across multiple GPUs: when ModelForge shards the model with FSDP.',
  lead:<>With more than one GPU there are two ways to share the work. Data parallel gives every GPU a full copy of the model and different batches. FSDP splits the model itself across GPUs, so together they can train a model none of them could hold alone.</>,
  sections:[
    {id:'compare',title:'The two schemes',body:<>
      <ParallelDiagram/>
      <Table head={['','Data parallel','FSDP']} mono={[]} rows={[
        [<b>Each GPU holds</b>,'The whole model, gradients and optimizer state','A shard of each, gathered layer by layer'],
        [<b>Speeds up</b>,'Throughput: more batches per second','Capacity: larger models'],
        [<b>Communication</b>,'Gradients once per step','Weights and gradients for every layer'],
        [<b>In ModelForge</b>,'Default for every multi-GPU run','Full fine-tuning only'],
      ]}/>
    </>},
    {id:'when',title:'When ModelForge picks FSDP',body:<>
      <p>The planner sums VRAM across GPUs when it sets the size ceiling, so the chosen model may be too big for any one card. It switches to FSDP when <b>all</b> of these hold:</p>
      <ul>
        <li>two or more CUDA GPUs are visible;</li>
        <li>the tier is full fine-tuning;</li>
        <li>the chosen model's budget, parameters × 18 bytes + 2 GB, exceeds the smallest GPU's memory.</li>
      </ul>
      <p>Otherwise it uses data parallel. Example: with two 24 GB GPUs, SFT has a ceiling of about 2.5B parameters. A 2B model needs about 38 GB, more than either card, so the plan selects FSDP.</p>
      <Callout tone="info">FSDP is never combined with LoRA or QLoRA. A request that tries is rejected.</Callout>
    </>},
    {id:'config',title:'What FSDP is configured with',body:<>
      <Code label="# applied to the training arguments">{`fsdp = "full_shard auto_wrap"
fsdp_config = {
  "min_num_params": 100_000_000,
  "backward_prefetch": "backward_pre",
  "state_dict_type": "FULL_STATE_DICT",
  "activation_checkpointing": true
}
gradient_checkpointing = false   # FSDP checkpoints activations itself
use_liger_kernel = false
activation_offloading = false`}</Code>
      <p>The final model is gathered into a single full state dict, so the output loads like any other model.</p>
    </>},
    {id:'launch',title:'How runs are launched',body:<p>ModelForge starts one training process per visible GPU through Accelerate. The console always trains on every GPU the server detects. To train on a subset, pass <C>gpu_indices</C> to <C>POST /api/recommendations</C> and <C>POST /api/runs</C> through the <D to="http-api">HTTP API</D>. DeepSpeed and tensor parallelism aren't supported.</p>},
    {id:'override',title:'Forcing a strategy',body:<>
      <p>The HTTP API accepts <C>parallelism_strategy_override</C> (<C>"data_parallel"</C> or <C>"fsdp"</C>) on <C>POST /api/runs</C>. The console doesn't expose it. See <D to="http-api">HTTP API</D>.</p>
    </>},
  ],
},
{
  slug:'gradient-checkpointing-and-packing',title:'Gradient checkpointing and packing',group:'Advanced',minutes:6,readFirst:'where-vram-goes',
  summary:'Trading compute for memory, packing short examples, and activation offloading.',
  lead:<>Two techniques make the most of each step. Gradient checkpointing saves memory by recomputing activations instead of storing them. Packing saves compute by filling each sequence with real tokens instead of padding.</>,
  sections:[
    {id:'checkpointing',title:'Gradient checkpointing',body:<>
      <p>The backward pass normally needs every layer's activations from the forward pass. Checkpointing keeps only some of them and recomputes the rest during backward. Training gets roughly 20–30% slower, and activation memory falls from growing with every layer to a small fraction of that.</p>
      <Table head={['Strategy','Gradient checkpointing']} rows={[
        ['sft, dpo, grpo','Always on'],
        ['qa','On below 12 GB of VRAM, off above'],
        ['Any run under FSDP','Replaced by FSDP activation checkpointing'],
      ]}/>
    </>},
    {id:'packing',title:'Packing',body:<>
      <p>Without packing, each training row holds one example padded out to <C>max_length</C>, and short examples waste most of the row. Packing concatenates several examples into each row.</p>
      <PackingDiagram/>
      <p>ModelForge enables packing for supervised training of causal LMs (Instruction Following and Text Generation, supervised) with the <C>bfd</C> (best-fit decreasing) strategy, which keeps examples whole. Summarization, DPO, GRPO and QA don't pack.</p>
    </>},
    {id:'lengths',title:'Sequence length',body:<>
      <p><C>max_length</C> caps the tokens per row. Longer examples are truncated. The plan sets it from total VRAM:</p>
      <Table head={['Total VRAM','SFT / DPO max_length','GRPO max_completion_length','QA max_length']} mono={[]} right={[1,2,3]} rows={[
        ['under 12 GB','512','128','256'],
        ['12–32 GB','1024','256','384'],
        ['32 GB or more','2048','512','384'],
      ]}/>
    </>},
    {id:'offload',title:'Activation offloading',body:<p>On GPUs with under 12 GB, SFT and DPO also move saved activations to CPU memory between the forward and backward passes. That costs some speed but lets small cards finish runs that would otherwise run out of memory.</p>},
  ],
},
{
  slug:'liger-kernels',title:'Liger kernels',group:'Advanced',minutes:3,
  summary:'Fused Triton kernels that reduce memory; how to install and when they are used.',
  lead:<>Liger is a set of fused GPU kernels for common transformer layers, such as RMSNorm, RoPE, SwiGLU and cross-entropy. They compute the same result with less memory and often a little faster, mostly by avoiding large intermediate tensors.</>,
  sections:[
    {id:'install',title:'Install',body:<>
      <Code>{`$ pip install "modelforge-finetuning[gpu]"`}</Code>
      <p>Liger requires CUDA and Triton. It has no CPU or Windows wheels, which is why it's an optional extra.</p>
    </>},
    {id:'when',title:'When ModelForge uses it',body:<Table head={['Strategy','use_liger_kernel']} rows={[
      ['sft, dpo','On automatically when a GPU is present and liger-kernel is installed'],
      ['grpo','Off'],
      ['qa, summarization','Not used'],
      ['Under FSDP','Off'],
    ]}/>},
    {id:'check',title:'Checking it',body:<p>Look for <C>use_liger_kernel: true</C> in the plan's <b>Training config</b>. If it reads <C>false</C> on a CUDA machine, the package isn't installed in the same environment as the server.</p>},
    {id:'compat',title:'Compatibility',body:<p>Liger patches specific architectures, including Llama, Mistral, Qwen, Gemma and Phi. Others train normally without the benefit. If a run fails inside a Liger kernel, uninstall <C>liger-kernel</C> and generate the plan again.</p>},
  ],
},
{
  slug:'resuming-from-checkpoints',title:'Resuming from checkpoints',group:'Advanced',minutes:4,
  summary:'Where checkpoints are written and how to resume a failed or cancelled run.',
  lead:<>Every 500 steps, a run saves a checkpoint holding the weights, optimizer state and scheduler position. If the run stops, you can continue from the last checkpoint instead of starting over.</>,
  sections:[
    {id:'where',title:'Where checkpoints go',body:<>
      <p>Checkpoints are written to <C>&lt;output dir&gt;/checkpoint-&lt;step&gt;</C> and listed on the run's detail page, newest first. All of them are kept, so long runs on large models can use a lot of disk. Delete old ones you don't need.</p>
    </>},
    {id:'resume',title:'Resuming',body:<>
      <ol>
        <li>Open the run under <b>Runs</b>.</li>
        <li>In <b>Checkpoints</b>, press <b>Resume</b> beside the step to continue from.</li>
        <li>ModelForge creates a new run with the same plan, dataset and GPUs, and continues from that step. You're taken to the new run.</li>
      </ol>
      <p>Resume appears on runs that failed or were cancelled, not on completed or active ones. Only one run can train at a time.</p>
    </>},
    {id:'restarts',title:'If the server stops',body:<p>When the server restarts, any run that was queued or running is marked failed with <i>“ModelForge stopped before this run reached a terminal state.”</i> Its checkpoints are intact, so resume it as above.</p>},
    {id:'playground',title:'Testing a checkpoint',body:<p>Any listed checkpoint can be loaded in the <D to="using-the-playground">Playground</D>, including from a run that's still training. That's a quick way to decide whether a run is worth finishing.</p>},
    {id:'cancel',title:'Cancelling',body:<p><b>Cancel run</b> stops after the current step. Steps since the last checkpoint are lost, so cancel shortly after a save if you plan to resume.</p>},
  ],
},
{
  slug:'fixing-out-of-memory-errors',title:'Fixing out-of-memory errors',group:'Advanced',minutes:5,readFirst:'where-vram-goes',
  summary:'Why runs still run out of memory and what to change.',
  lead:<>The plan is an estimate, not a guarantee. If a run fails with <C>CUDA out of memory</C>, the run's error shows the message and you can try again with a smaller footprint. ModelForge doesn't retry automatically.</>,
  sections:[
    {id:'causes',title:'Common causes',body:<ul>
      <li><b>Something else is using the GPU</b>: a desktop session, a notebook, another training job, or a model loaded elsewhere. The planner counts total memory, not free memory.</li>
      <li><b>Long examples.</b> The activation budget assumes rows of up to <C>max_length</C>. Few models fail at the default lengths, but large models near the ceiling can.</li>
      <li><b>DPO and GRPO</b> hold extra copies or completions in memory and are more sensitive than SFT.</li>
      <li><b>Uneven GPUs.</b> Under data parallel, every GPU must hold the whole model, including the smallest one.</li>
    </ul>},
    {id:'fixes',title:'What to change, in order',body:<Table head={['Fix','How']} mono={[]} rows={[
      [<b>Free the GPU</b>,<>Close other GPU processes, then check with <C>nvidia-smi</C>.</>],
      [<b>Pick a smaller model</b>,'Choose a lower row in the recommendation table, or type in a smaller model ID.'],
      [<b>Install Liger kernels</b>,<>See <D to="liger-kernels">Liger kernels</D>. They can cut activation memory substantially.</>],
      [<b>Shorten examples</b>,'Trim or split very long rows in your dataset.'],
      [<b>Add GPUs</b>,'With two or more, large full fine-tunes switch to FSDP automatically.'],
      [<b>Resume</b>,<>If it failed partway through, fix the cause and <D to="resuming-from-checkpoints">resume from the last checkpoint</D>.</>],
    ]}/>},
    {id:'playground',title:'Out of memory in the Playground',body:<p>The Playground checks memory before loading and refuses requests it can't fit. Choose a smaller context window, shorten the conversation, or set <C>MODELFORGE_OFFLOAD_DIR</C> to a drive with more free space.</p>},
    {id:'api',title:'Changing hyperparameters',body:<p>The console shows the training config read-only. Through the <D to="http-api">HTTP API</D> you can send an edited <C>recommendation</C> to <C>POST /api/runs</C>, for example with a smaller <C>max_length</C>. That's used as sent when <C>model_id</C> is the recommendation's top candidate.</p>},
  ],
},
];
