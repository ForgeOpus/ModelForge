import {C, Callout, Code, D, Table, type DocPage} from '../ui';
import {LoraDiagram, TierFlow, VramBars} from '../diagrams';

export const intermediate:DocPage[]=[
{
  slug:'where-vram-goes',title:'Where VRAM goes',group:'Intermediate',minutes:7,readFirst:'what-fine-tuning-is',
  summary:'Weights, gradients, optimizer state and activations; how ModelForge budgets memory per strategy.',
  lead:<>Training needs far more memory than running a model. Besides the weights themselves, the GPU has to hold gradients, optimizer state and the activations of every layer. ModelForge turns that into a simple budget so it can tell you what fits before you start.</>,
  tryIt:{text:'Generate a plan and read its VRAM estimate.',to:'/app',label:'Open in console'},
  sections:[
    {id:'parts',title:'The four parts',body:<Table head={['Part','What it is','Scales with']} mono={[]} rows={[
      [<b>Weights</b>,'The model itself.','Parameter count × bytes per weight'],
      [<b>Gradients</b>,'One value per trained weight, computed each step.','Trained parameters'],
      [<b>Optimizer state</b>,'Adam keeps two running averages per trained weight.','Trained parameters'],
      [<b>Activations</b>,'Intermediate results kept for the backward pass.','Batch size × sequence length × layers'],
    ]}/>},
    {id:'budget',title:'How ModelForge budgets it',body:<>
      <p>Rather than simulate every allocation, the planner uses a fixed cost per parameter for each <D to="how-a-tier-is-chosen">tier</D>, plus two constants:</p>
      <ul>
        <li><b>2.0 GB</b> activation budget for one sequence at a time.</li>
        <li><b>0.1 GB</b> for LoRA adapters and their optimizer state.</li>
      </ul>
      <VramBars/>
      <p>Full fine-tuning is costed at 18 bytes per parameter because it stores gradients and optimizer state for every weight. Adapter tiers pay only for frozen weights at 2, 1 or 0.5 bytes.</p>
    </>},
    {id:'adjusted',title:'Adjusting for the strategy',body:<>
      <p>Some objectives need more than one model or more than one sequence in memory. The planner first converts your raw VRAM into an <b>adjusted</b> figure:</p>
      <Code>{`adjusted = (raw VRAM − (batch_multiplier − 1) × 2.0 GB) / (1 + extra_model_copies)`}</Code>
      <Table head={['Strategy','Extra model copies','Batch multiplier','Why']} mono={[0]} right={[1,2]} rows={[
        ['sft','0','1','One model, one sequence.'],
        ['dpo','1','2','A frozen reference model, and chosen and rejected are processed together.'],
        ['grpo','0','4','Four sampled completions per prompt.'],
        ['qa','0','1','One model, one window.'],
      ]}/>
    </>},
    {id:'example',title:'A worked example',body:<>
      <p>On a single 24 GB GPU, the largest model that fits at full fine-tuning is:</p>
      <Table head={['Strategy','Adjusted VRAM','Largest model']} mono={[0]} right={[1,2]} rows={[
        ['sft','24 GB','≈ 1.2B parameters'],
        ['grpo','18 GB','≈ 0.9B parameters'],
        ['dpo','11 GB','≈ 0.5B parameters'],
      ]}/>
      <p>For SFT that's (24 − 2) ÷ 18 ≈ 1.22 billion. The plan's <b>VRAM estimate</b> panel shows the same arithmetic for your machine.</p>
    </>},
    {id:'caveats',title:'What the budget leaves out',body:<>
      <p>The 2 GB activation budget assumes the plan's batch size of 1 with gradient checkpointing on. Very long sequences, other processes on the GPU, or a desktop session on the same card can still push a run over. See <D to="fixing-out-of-memory-errors">Fixing out-of-memory errors</D>.</p>
      <Callout tone="info" title="Multiple GPUs are summed.">The planner adds the VRAM of every visible GPU together. That's only safe when the model is sharded. See <D to="data-parallel-vs-fsdp">Data parallel vs. FSDP</D> for when ModelForge does that.</Callout>
    </>},
  ],
},
{
  slug:'full-fine-tuning',title:'Full fine-tuning',group:'Intermediate',minutes:4,readFirst:'where-vram-goes',
  summary:'Updating every weight: cost, precision, outputs and when it is chosen.',
  lead:<>Full fine-tuning updates every weight in the model. It gives the best results for a given base model and is what ModelForge picks whenever a model fits, but it costs about nine times more memory per parameter than LoRA.</>,
  sections:[
    {id:'cost',title:'What it costs',body:<p>Each parameter needs its weight, its gradient and two Adam optimizer states. ModelForge budgets this at <b>18 bytes per parameter</b> plus 2 GB for activations. A 1B-parameter model needs about 20 GB. A 7B model needs about 128 GB, which means several GPUs with <D to="data-parallel-vs-fsdp">FSDP</D>.</p>},
    {id:'when',title:'When ModelForge uses it',body:<>
      <p>The planner tries tiers from highest to lowest quality and stops at the first one with room for a model. Full fine-tuning is first, so on any CUDA GPU with more than 2 GB of adjusted VRAM the plan says <C>full_finetune</C>. The recommended models are then the ones small enough to train that way.</p>
      <p>If you want a larger model than full fine-tuning allows, use more GPUs. The planner sums their memory and switches to FSDP when a single GPU can't hold the model.</p>
    </>},
    {id:'precision',title:'Precision',body:<p>Training runs in <C>bf16</C> when every GPU supports it (compute capability 8.0 or newer, such as Ampere, Ada and Hopper), and in <C>fp16</C> otherwise. CPU and Apple MPS runs use full precision.</p>},
    {id:'outputs',title:'What gets saved',body:<p>A complete copy of the fine-tuned model in the output directory, ready to load with <C>from_pretrained</C>. That's as large as the base model, unlike a LoRA run, which saves only the adapter.</p>},
    {id:'dpo-note',title:'With DPO',body:<p>Full fine-tuning with DPO loads a second, frozen copy of the model as the reference. That's why the planner halves the adjusted VRAM for preference training.</p>},
  ],
},
{
  slug:'lora-and-qlora',title:'LoRA and QLoRA',group:'Intermediate',minutes:9,readFirst:'where-vram-goes',
  summary:'Low-rank adapters, quantized base weights, ranks, alpha and target modules.',
  lead:<>LoRA freezes the base model and trains a small pair of matrices next to selected layers. QLoRA does the same thing on top of a base model that has been quantized to 8 or 4 bits. Both give up a little quality in exchange for training much larger models in the same memory.</>,
  tryIt:{text:'Generate a plan and compare its PEFT decision with this page.',to:'/app',label:'Open in console'},
  sections:[
    {id:'idea',title:'The idea',body:<>
      <p>Full fine-tuning updates every weight matrix <C>W</C> in the model. LoRA leaves <C>W</C> frozen and learns a low-rank update instead: two thin matrices, <C>B</C> and <C>A</C>, whose product has the same shape as <C>W</C>. The rank <C>r</C> sets how thin they are.</p>
      <LoraDiagram/>
      <p>Because only <C>A</C> and <C>B</C> get gradients and optimizer state, the expensive part of training memory almost disappears. The frozen weights still have to sit on the GPU, which is where QLoRA comes in: it stores them in 8-bit or 4-bit form and dequantizes them on the fly.</p>
    </>},
    {id:'cost',title:'What it costs in memory',body:<>
      <p>ModelForge budgets each tier with a fixed cost per parameter, plus 0.1 GB for adapters and 2.0 GB for activations.</p>
      <Table head={['Tier','Bytes / param','What sits on the GPU','7B model weights']} mono={[]} right={[1,3]} rows={[
        [<b>Full fine-tune</b>,'18','Weights, gradients and Adam optimizer state','126 GB'],
        [<b>LoRA bf16</b>,'2','Frozen bf16 weights + small adapters','14 GB'],
        [<b>QLoRA 8-bit</b>,'1','Frozen 8-bit weights + adapters','7 GB'],
        [<b>QLoRA 4-bit</b>,'0.5','Frozen 4-bit weights + adapters','3.5 GB'],
      ]}/>
    </>},
    {id:'when',title:'When ModelForge picks it',body:<Callout title="When ModelForge picks it.">The engine takes the highest-fidelity tier whose ceiling is above zero. On any CUDA GPU with more than about 2 GB, that's full fine-tuning of a smaller model. Adapters are used on Apple MPS when full fine-tuning doesn't fit (LoRA bf16), and on CPU-only machines, which train an adapter on unquantized weights. The console has no control to force a lower tier yet.</Callout>},
    {id:'configure',title:'Configure it',body:<>
      <p>The recommendation's <C>peft_config</C> holds the adapter settings. This is what ModelForge generates for a 3B model:</p>
      <Code label="# recommendation.peft_config">{`{
  "r": 16,
  "lora_alpha": 32,
  "target_modules": "all-linear"
}`}</Code>
      <Table head={['Key','ModelForge default','What it controls']} rows={[
        ['r','SFT and DPO: 8 below 1B · 16 below 7B · 32 below 13B · 64 above. GRPO: 8 below 1B, 16 above. QA: 16 below 7B, 32 above.','Rank of the update. Higher learns more, costs more.'],
        ['lora_alpha','2 × r','Scales the update by α / r, so 2 at the defaults.'],
        ['target_modules','"all-linear"','Every linear layer in the model gets an adapter, attention and MLP alike.'],
        ['load_in_8bit / 4bit','Set by the QLoRA tier','Quantizes the frozen base weights at load time.'],
      ]}/>
    </>},
    {id:'outputs',title:'What gets saved',body:<p>Only the adapter, typically a few megabytes to a few hundred, plus <C>modelforge_config.json</C> naming the base model. Load it with PEFT on top of the same base model. ModelForge doesn't merge the adapter into the base weights.</p>},
    {id:'rank',title:'Advanced: choosing a rank',body:<p>Start with the default. Raise <C>r</C> if training loss plateaus well above where full fine-tuning would get, and lower it if the model overfits a small dataset. Because ModelForge already targets every linear layer, rank is the main capacity knob.</p>},
  ],
},
{
  slug:'how-a-tier-is-chosen',title:'How a tier is chosen',group:'Intermediate',minutes:5,readFirst:'lora-and-qlora',
  summary:'The parameter ceiling per tier, selection order, CPU and MPS rules, and plan warnings.',
  lead:<>The tier decides how many bytes each parameter costs during training, which in turn decides how large a model you can train. ModelForge computes a parameter ceiling for every tier and takes the best one with room to spare.</>,
  tryIt:{text:'Generate a plan and open the PEFT decision.',to:'/app',label:'Open in console'},
  sections:[
    {id:'flow',title:'The decision',body:<TierFlow/>},
    {id:'ceilings',title:'Parameter ceilings',body:<>
      <p>For each tier the largest trainable model is:</p>
      <Table head={['Tier','Ceiling (billions of parameters)']} rows={[
        ['full_finetune','(adjusted − 2.0) / 18'],
        ['lora_bf16','(adjusted − 2.1) / 2'],
        ['qlora_8bit','(adjusted − 2.1) / 1'],
        ['qlora_4bit','(adjusted − 2.1) / 0.5'],
      ]}/>
      <p>Adjusted VRAM comes from <D to="where-vram-goes">Where VRAM goes</D>. The selection order is full, then LoRA bf16, 8-bit, 4-bit, and the first ceiling above zero wins. That ceiling then filters the <D to="picking-a-base-model">model search</D>.</p>
    </>},
    {id:'special',title:'CPU and Apple Silicon',body:<Table head={['Hardware','Tier','Ceiling']} mono={[]} rows={[
      [<b>CPU only</b>,<C>qlora_4bit</C>,'The smaller of 500M parameters and RAM ÷ 3 bytes. Quantization is switched off, so it trains an adapter on full-precision weights. Expect it to be very slow.'],
      [<b>Apple MPS</b>,<><C>full_finetune</C>, else <C>lora_bf16</C></>,'MPS memory is counted as 8 GB. bitsandbytes doesn\'t support MPS, so the 8-bit and 4-bit tiers are disabled.'],
      [<b>CUDA</b>,'First ceiling above zero','Full fine-tuning whenever adjusted VRAM is above 2 GB.'],
    ]}/>},
    {id:'read-plan',title:'Reading the plan',body:<>
      <p>The plan's <b>PEFT decision</b> names the tier and why. <b>VRAM estimate</b> shows raw and adjusted VRAM and the ceilings. Warnings appear below. The ones to act on:</p>
      <ul>
        <li><b>No single-GPU tier is viable.</b> Even QLoRA 4-bit doesn't fit. Free GPU memory or add GPUs.</li>
        <li><b>No models found on the Hub.</b> Nothing under the ceiling matched the use case, so no training config was generated. Add VRAM or choose a different use case.</li>
        <li><b>Available disk space may be insufficient.</b> The model's weights may not fit on disk, before counting checkpoints.</li>
        <li><b>Multi-GPU.</b> States whether training will use data parallel or FSDP.</li>
      </ul>
    </>},
  ],
},
{
  slug:'picking-a-base-model',title:'Picking a base model',group:'Intermediate',minutes:4,readFirst:'how-a-tier-is-chosen',
  summary:'How recommendations are searched and ranked, and using your own model ID.',
  lead:<>Once the tier sets a size ceiling, ModelForge searches the Hugging Face Hub for popular models of the right type that fit under it. You can take its top pick, choose another from the list, or type in any model ID.</>,
  tryIt:{text:'Generate a recommendation and compare the candidates.',to:'/app',label:'Open in console'},
  sections:[
    {id:'search',title:'How the search works',body:<ol>
      <li>Query the Hub for the use case's pipeline tag, sorted by downloads.</li>
      <li>Skip models that don't publish a parameter count in their safetensors metadata. The plan says how many were skipped.</li>
      <li>Keep models at or under the tier's ceiling, up to five.</li>
      <li>Build the training config for the top result, the most downloaded.</li>
    </ol>},
    {id:'choosing',title:'Choosing from the list',body:<p>The table shows each candidate's parameters and downloads. Downloads measure popularity, not quality. When two candidates are close, prefer the newer model, or the instruct variant for instruction following. Picking a model other than the top one rebuilds the config for that model when the run starts.</p>},
    {id:'custom',title:'Using your own model ID',body:<>
      <p>Type any Hub ID into <b>Or enter a different model ID</b>. It must:</p>
      <ul>
        <li>exist and be readable with your Hugging Face login (gated models need their licence accepted);</li>
        <li>have the pipeline tag the use case expects;</li>
        <li>publish its parameter count, and fit under the plan's ceiling.</li>
      </ul>
      <p>Otherwise <b>Start training</b> is refused with <i>“Cannot safely configure this model without valid parameter metadata”</i>, <i>“Selected model exceeds the recommendation's memory budget”</i>, or a message naming the incompatible pipeline tag.</p>
    </>},
    {id:'tips',title:'Tips',body:<ul>
      <li>Smaller is often fine. A 0.5B model fine-tuned on good data beats a larger one fine-tuned on noise, and trains several times faster.</li>
      <li>For instruction following, start from a model that is already chat-tuned.</li>
      <li>Rerun <b>Generate recommendation</b> after changing GPUs. Plans are computed from the hardware at that moment.</li>
    </ul>},
  ],
},
];
