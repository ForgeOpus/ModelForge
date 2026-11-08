import {C, Callout, Cards, Code, D, Steps, Table, type DocPage} from '../ui';

export const getStarted:DocPage[]=[
{
  slug:'introduction',title:'Introduction',group:'Get started',minutes:3,
  summary:'What ModelForge does and where to start reading.',
  lead:<>ModelForge fine-tunes open models from the Hugging Face Hub on the machine you run it on. It reads your GPUs, works out how large a model fits, picks a training recipe for your goal, and runs it while you watch the loss come down.</>,
  sections:[
    {id:'what',title:'What it does for you',body:<>
      <p>Fine-tuning normally means choosing a base model, a precision, an adapter configuration, a batch size and a parallelism scheme, then discovering at step 40 that it runs out of memory. ModelForge makes those choices from your hardware and shows its reasoning before anything starts.</p>
      <ul>
        <li><b>Hardware-aware planning.</b> Total VRAM, the strategy's memory overhead and a fixed per-parameter cost give a ceiling on model size. See <D to="where-vram-goes">Where VRAM goes</D>.</li>
        <li><b>A recipe per goal.</b> Supervised fine-tuning, DPO preference training and GRPO reinforcement training, plus native trainers for summarization and extractive QA.</li>
        <li><b>Live runs.</b> Streamed training logs, checkpoints you can resume from, and TensorBoard logging.</li>
        <li><b>A playground.</b> Chat with a finished model or any checkpoint without leaving the browser.</li>
      </ul>
    </>},
    {id:'surfaces',title:'Three ways to use it',body:<>
      <p>The browser console, the terminal wizard and notebooks all call the same planning, dataset and training code, so a dataset that validates in one will train in the others.</p>
      <Table head={['Surface','Start it with','Best for']} rows={[
        ['Browser console','modelforge run','Most people. Everything in these docs refers to it unless noted.'],
        ['Terminal wizard','modelforge cli','Headless servers over SSH.'],
        ['Notebook','modelforge-nb','Jupyter and Colab sessions.'],
      ]}/>
    </>},
    {id:'next',title:'Where to go next',body:<Cards items={[
      {to:'install-and-first-run',title:'Install and first run',text:'Get the server running and connect your browser.'},
      {to:'what-fine-tuning-is',title:'What fine-tuning is',text:'New to this? Start with the concepts.'},
      {to:'preparing-a-dataset',title:'Preparing a dataset',text:'Column names and file formats for every use case.'},
      {to:'how-a-tier-is-chosen',title:'How a tier is chosen',text:'Understand the plan before you train.'},
    ]}/>},
  ],
},
{
  slug:'install-and-first-run',title:'Install and first run',group:'Get started',minutes:4,
  summary:'Install the package, start the server and open the console with its token.',
  lead:<>Install the Python package, start the server, and open the link it prints. The link carries a one-time token that connects your browser.</>,
  tryIt:{text:'Server already running? Open the console and start a run.',to:'/app',label:'Open console'},
  sections:[
    {id:'requirements',title:'Requirements',body:<>
      <ul>
        <li><b>Python 3.13</b> or newer.</li>
        <li>An NVIDIA GPU with CUDA for real training. Apple Silicon (MPS) and CPU-only machines work, but only with small models. See <D to="how-a-tier-is-chosen">How a tier is chosen</D>.</li>
        <li>Disk space for base models, checkpoints and outputs. The plan warns when free space looks too small.</li>
      </ul>
    </>},
    {id:'install',title:'Install',body:<>
      <Code>{`$ pip install modelforge-finetuning`}</Code>
      <p>Optional extras:</p>
      <Table head={['Extra','Adds','When to use it']} rows={[
        ['[cli]','click, rich','The terminal wizard (modelforge cli).'],
        ['[gpu]','liger-kernel','Fused kernels that cut memory on CUDA. No CPU or Windows wheels. See Liger kernels.'],
      ]}/>
      <Code>{`$ pip install "modelforge-finetuning[cli,gpu]"`}</Code>
    </>},
    {id:'start',title:'Start the server',body:<>
      <Code>{`$ modelforge run
ModelForge v4 browser UI
Serving on: http://0.0.0.0:8000
Open:       http://127.0.0.1:8000/app?token=…
The startup token is required for API and WebSocket access.`}</Code>
      <p>Your browser opens the <b>Open</b> link automatically. The console stores the token, removes it from the address bar, and sends it with every request. A new token is generated each time the server starts.</p>
      <Table head={['Flag','Default','Effect']} rows={[
        ['--host','0.0.0.0','Interface to bind.'],
        ['--port','8000','Port to bind.'],
        ['--no-browser','off','Print the link without opening a browser. Use this on remote machines.'],
        ['--token','random','Use a fixed token so bookmarks keep working across restarts.'],
      ]}/>
      <Callout tone="warn" title="The server listens on every interface by default.">Anyone on your network who has the token can start runs and read your data. Pass <C>--host 127.0.0.1</C> on shared networks, and reach a remote machine through an SSH tunnel.</Callout>
    </>},
    {id:'connect',title:'Connecting another browser',body:<>
      <p>Open the printed link in the new browser, or paste the token under <b>Settings → Connection → API token</b> and reload. If the console says the connection is unavailable, the token is missing or out of date.</p>
    </>},
    {id:'hub-login',title:'Hugging Face login',body:<>
      <p>Public models and datasets need no login. For gated models such as Llama, accept the licence on the model's Hub page, then log in once on the machine running the server:</p>
      <Code>{`$ huggingface-cli login`}</Code>
      <p>The server uses this cached login whenever it downloads from the Hub.</p>
    </>},
    {id:'where-data-lives',title:'Where things are stored',body:<Table head={['What','Where']} mono={[]} rows={[
      ['Run history, checkpoints list, model registry',<><C>modelforge.db</C> in your user data folder (<C>~/.local/share/ModelForge</C> on Linux)</>],
      ['Trained models and checkpoints','The output directory you choose for each run (default ./modelforge_output)'],
      ['TensorBoard logs',<><C>&lt;output dir&gt;/tensorboard</C></>],
      ['Downloaded base models','The standard Hugging Face cache'],
    ]}/>},
  ],
},
{
  slug:'tour-of-the-console',title:'Tour of the console',group:'Get started',minutes:5,readFirst:'install-and-first-run',
  summary:'New run, Runs, Playground, Artifacts and Settings pages explained.',
  lead:<>The console has six pages in its sidebar. Most of your time is spent on <b>New run</b>, where you plan a fine-tune, and on a run's detail page, where you watch it.</>,
  tryIt:{text:'Follow along by opening a new run.',to:'/app',label:'Open in console'},
  sections:[
    {id:'new-run',title:'New run',body:<>
      <p>The left side is a four-step form. The right side is the <b>Plan</b>, which fills in once you generate a recommendation.</p>
      <Steps items={[
        ['Task',<p>Pick a use case tile, then an objective: <b>Supervised</b>, <b>Preference</b> or <b>Reinforcement</b>. Objectives the use case can't train are greyed out.</p>],
        ['Dataset',<p>Enter a local path or a Hub dataset ID and switch between <b>Local</b> and <b>Hub</b>. <b>Validate</b> checks the columns against the objective. It's optional, but it catches problems before you spend GPU time.</p>],
        ['Base model',<p><b>Generate recommendation</b> asks the server for up to five models that fit. Pick one from the table, or type any Hub model ID into <b>Or enter a different model ID</b>.</p>],
        ['Output',<p>Choose the output directory and whether to write TensorBoard logs, then <b>Start training</b>.</p>],
      ]}/>
      <p>The Plan shows the chosen tier and model, the strategy and parallelism, a <b>VRAM estimate</b>, the <b>PEFT decision</b>, the full <b>Training config</b> that will be used, and any warnings. The config is read-only in the console. See <D to="training-config-keys">Training config keys</D> for what each value means.</p>
      <Callout tone="info" title="Give each run its own output directory.">Every run defaults to <C>./modelforge_output</C>. Two runs writing to the same folder overwrite each other's weights and checkpoints.</Callout>
    </>},
    {id:'runs',title:'Runs and run detail',body:<>
      <p><b>Runs</b> lists every run with its status. Open one to see:</p>
      <ul>
        <li><b>Training log.</b> Status, loss and step updates streamed live while the run is active.</li>
        <li><b>Checkpoints.</b> Each saved step, with a <b>Resume</b> button on runs that failed or were cancelled. See <D to="resuming-from-checkpoints">Resuming from checkpoints</D>.</li>
        <li><b>Run details.</b> Model, use case, objective, dataset and output directory.</li>
        <li><b>Cancel run.</b> Stops the run after the current step.</li>
      </ul>
      <p>Only one run trains at a time. Starting another while one is active is refused.</p>
    </>},
    {id:'playground',title:'Playground',body:<p>Chat with a completed run's final model or any of its checkpoints. See <D to="using-the-playground">Using the Playground</D>.</p>},
    {id:'artifacts',title:'Artifacts',body:<p>Completed runs, with the base model, use case and output directory of each. Each output folder contains the trained weights (or just the adapter for LoRA runs), the checkpoints, TensorBoard logs and a <C>modelforge_config.json</C> describing how the model was trained.</p>},
    {id:'settings',title:'Settings',body:<p>Theme (Light, Dark or System), the API token for this browser, and read-outs of the hardware and environment the server detected. If the hardware here looks wrong, the plan will be wrong too.</p>},
  ],
},
{
  slug:'using-the-playground',title:'Using the Playground',group:'Get started',minutes:4,readFirst:'tour-of-the-console',
  summary:'Chat with a trained model or checkpoint; context windows, memory and offload.',
  lead:<>The Playground loads a trained model for each message you send, generates a reply, and unloads it again. It can spread a model across GPU, CPU memory and disk, so it works even while a run is training.</>,
  tryIt:{text:'Pick a completed run and send it a message.',to:'/app/playground',label:'Open Playground'},
  sections:[
    {id:'choose',title:'Choosing what to talk to',body:<>
      <Table head={['Control','What it does']} mono={[]} rows={[
        ['Run','Any run. Completed runs offer their final model.'],
        ['Model output',<><b>Final model</b> for completed runs, or a checkpoint listed as <b>Step N</b>. Checkpoints work on runs that are still training or that failed.</>],
        ['Context window','1,024, 4,096 (default), 8,016 or 32,768 tokens.'],
      ]}/>
      <p>Changing the run or checkpoint starts a fresh conversation.</p>
    </>},
    {id:'context',title:'Context windows',body:<>
      <p>The effective limit is the smallest of the window you choose, the model's own maximum, and its tokenizer's limit. Every reply reserves 256 tokens for output. If the conversation and the reply don't fit, the request is refused with an error. Earlier messages are never silently dropped.</p>
      <p>Below each reply the console shows the input token count, the limit that applied, and the device used.</p>
    </>},
    {id:'memory',title:'Memory and offload',body:<>
      <p>Weights are loaded at full float32 precision and placed across devices with Accelerate. When no run is training, the Playground can use free GPU memory, then CPU RAM, then disk. While a run is training, it never touches the GPUs, so the run keeps its VRAM.</p>
      <ul>
        <li>It keeps back at least 2 GB of RAM (8 GB while training) for the system and the training job.</li>
        <li>Disk offload goes to a temporary folder that is deleted afterwards. Set <C>MODELFORGE_OFFLOAD_DIR</C> to put it on a faster or larger drive.</li>
        <li>If the model can't fit anywhere, the request is refused rather than starting and crashing.</li>
      </ul>
      <Callout tone="warn" title="Disk offload is slow.">A reply can take minutes when most of the model lives on disk. CPU inference also competes with a CPU-bound training run.</Callout>
    </>},
    {id:'qa',title:'Extractive QA models',body:<>
      <p>Send a JSON message with a question and a context. The reply is a span copied out of the context.</p>
      <Code>{`{"question": "Where is Paris?", "context": "Paris is in France."}`}</Code>
    </>},
    {id:'privacy',title:'Conversations',body:<p>Conversations exist only in this browser tab. <b>New chat</b>, switching models or leaving the page discards them, and the server stores nothing. Use <b>Export Markdown</b> to keep a copy. Replies are greedy (no sampling), so the same conversation gives the same answer.</p>},
  ],
},
];
