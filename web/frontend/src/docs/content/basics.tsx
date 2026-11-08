import {C, Callout, Code, D, Table, type DocPage} from '../ui';

export const basics:DocPage[]=[
{
  slug:'what-fine-tuning-is',title:'What fine-tuning is',group:'Basics',minutes:4,
  summary:'Pretraining vs fine-tuning, what changes in the model, and when it helps.',
  lead:<>A pretrained model already knows language. Fine-tuning keeps training it on a small, focused dataset so it picks up a particular behaviour: following your instructions, writing in your format, or answering from a document.</>,
  sections:[
    {id:'pretraining',title:'Pretraining and fine-tuning',body:<>
      <p>Base models on the Hub are pretrained on huge amounts of text to predict the next token. That gives them grammar, facts and reasoning, but not your task. Fine-tuning continues training from those weights with a learning rate small enough that the model adapts without forgetting what it knows.</p>
      <p>Because the starting point is already capable, a few thousand good examples often matter more than millions of mediocre ones.</p>
    </>},
    {id:'what-changes',title:'What actually changes',body:<>
      <p>Training adjusts the model's weights to make your examples more likely. There are two ways to do that:</p>
      <ul>
        <li><b>Full fine-tuning</b> updates every weight. This gives the highest quality but needs the most memory. See <D to="full-fine-tuning">Full fine-tuning</D>.</li>
        <li><b>Parameter-efficient fine-tuning (PEFT)</b> freezes the model and trains a small adapter beside it. See <D to="lora-and-qlora">LoRA and QLoRA</D>.</li>
      </ul>
      <p>ModelForge decides which to use from your hardware. The plan names the choice under <b>PEFT decision</b>.</p>
    </>},
    {id:'when',title:'When fine-tuning helps',body:<Table head={['Good fit','Poor fit']} mono={[]} rows={[
      ['A consistent format or tone across outputs','Teaching new facts that change often (use retrieval instead)'],
      ['A narrow task the base model half-does','Tasks a good prompt already solves'],
      ['Preferring one style of answer over another','Very small datasets of a few dozen rows'],
      ['Extracting answers from passages','Making a small model as capable as a much larger one'],
    ]}/>},
    {id:'loop',title:'The loop in ModelForge',body:<>
      <ol>
        <li>Pick what you want the model to do: a <D to="use-cases-and-tasks">use case</D> and an <D to="objectives">objective</D>.</li>
        <li><D to="preparing-a-dataset">Prepare a dataset</D> with the columns that objective needs.</li>
        <li>Generate a plan and start training.</li>
        <li>Try the result in the <D to="using-the-playground">Playground</D>, then adjust the data and train again.</li>
      </ol>
    </>},
  ],
},
{
  slug:'objectives',title:'Objectives: SFT, DPO and GRPO',group:'Basics',minutes:6,readFirst:'what-fine-tuning-is',
  summary:'Supervised, preference (DPO) and reinforcement (GRPO) objectives and their data.',
  lead:<>The objective is how the model learns. Supervised training copies example answers, preference training learns which of two answers is better, and reinforcement training rewards answers that are correct.</>,
  tryIt:{text:'Pick a use case and see which objectives it supports.',to:'/app',label:'Open in console'},
  sections:[
    {id:'overview',title:'At a glance',body:<Table head={['Objective','Strategy','You provide','The model learns']} mono={[1]} rows={[
      ['Supervised','sft','prompt → completion','To produce the completion for each prompt.'],
      ['Preference','dpo','chosen and rejected answers','To prefer chosen over rejected, without a separate reward model.'],
      ['Reinforcement','grpo','prompt and a reference answer','To generate answers that match the reference.'],
    ]}/>},
    {id:'sft',title:'Supervised fine-tuning',body:<>
      <p>The standard starting point. Each row pairs an input with the output you want, and training minimises the loss on the output tokens. Use it to teach a format, a task or a voice.</p>
      <Code label="instruction_following · supervised">{`{"prompt": "Summarise this ticket: …", "completion": "Customer cannot log in after the update."}`}</Code>
      <p>The plan uses a learning rate of <C>2e-5</C>, packs short examples together (see <D to="gradient-checkpointing-and-packing">packing</D>), and trains for 3 epochs.</p>
    </>},
    {id:'dpo',title:'Preference training (DPO)',body:<>
      <p>Direct Preference Optimization learns from pairs of answers where one is better. It nudges the model towards the chosen answer and away from the rejected one, relative to a frozen reference copy of the starting model. Use it after SFT to polish style, safety or helpfulness.</p>
      <Code label="instruction_following · preference">{`{"prompt": "Explain LoRA", "chosen": "LoRA trains two small matrices…", "rejected": "It is a thing."}`}</Code>
      <p><C>prompt</C> is optional when each answer already contains its own context. ModelForge uses <C>beta = 0.1</C> and the sigmoid loss, with very small learning rates (<C>1e-7</C> for full fine-tuning, up to <C>5e-6</C> for LoRA on small models).</p>
      <Callout tone="info" title="DPO needs more memory.">Full fine-tuning keeps a second, frozen copy of the model as the reference, and every step processes both answers. The planner halves the usable VRAM for DPO. See <D to="where-vram-goes">Where VRAM goes</D>.</Callout>
    </>},
    {id:'grpo',title:'Reinforcement training (GRPO)',body:<>
      <p>Group Relative Policy Optimization samples several answers per prompt (4 in ModelForge), scores each one, and pushes the model towards the higher-scoring ones. The score comes from a reward function instead of an example answer.</p>
      <Code label="text_generation · reinforcement">{`{"prompt": "What is 17 × 3? Reply with the number only.", "answer": "51"}`}</Code>
      <Callout tone="warn" title="The built-in reward is exact match.">A completion scores 1 when it equals the reference answer after trimming surrounding whitespace, and 0 otherwise. Use GRPO for tasks with one checkable answer, such as arithmetic, classification labels or short extractions. It won't reward general helpfulness, and custom reward functions aren't supported yet.</Callout>
    </>},
    {id:'choosing',title:'Which should I use?',body:<ul>
      <li>Start with <b>Supervised</b>. It's the most forgiving and the cheapest.</li>
      <li>Add <b>Preference</b> when outputs are correct but you prefer some over others.</li>
      <li>Use <b>Reinforcement</b> when you can't write the ideal answer, but you can check whether an answer is right.</li>
    </ul>},
  ],
},
{
  slug:'use-cases-and-tasks',title:'Use cases and tasks',group:'Basics',minutes:4,
  summary:'The four use cases, their model heads, trainers and supported objectives.',
  lead:<>A use case is the goal in plain terms. Choosing one fixes which kind of model ModelForge searches for, which head it loads, and which objectives it can train.</>,
  sections:[
    {id:'table',title:'Available use cases',body:<Table head={['Use case','Model type','Objectives','Trainer']} mono={[]} rows={[
      [<b>Instruction Following</b>,'Causal LM (text-generation)','Supervised, Preference, Reinforcement','SFT, DPO or GRPO'],
      [<b>Text Generation</b>,'Causal LM (text-generation)','Supervised, Preference, Reinforcement','SFT, DPO or GRPO'],
      [<b>Summarization</b>,'Encoder-decoder (summarization)','Supervised','Seq2Seq trainer'],
      [<b>Extractive QA</b>,'Encoder with QA head (question-answering)','Supervised','QA trainer with start/end labels'],
    ]}/>},
    {id:'instruction-vs-text',title:'Instruction following vs. text generation',body:<p>Both train the same kind of model. They differ in how your columns are described and which aliases they accept. Instruction following speaks in instructions and responses, text generation in input text and continuations. Pick whichever matches how your data is already labelled. See <D to="preparing-a-dataset">Preparing a dataset</D>.</p>},
    {id:'summarization',title:'Summarization',body:<p>Uses a sequence-to-sequence model such as T5 or BART, trained with a native encoder-decoder trainer rather than a chat-style one. Your <C>document</C> is the input and <C>summary</C> is the target. Both are truncated to the plan's <C>max_length</C>.</p>},
    {id:'qa',title:'Extractive QA',body:<p>The model doesn't write an answer. It points at the start and end of the answer inside a passage. Long passages are split into overlapping windows (384 tokens, overlapping by 128), so answers near a boundary aren't lost. The model's tokenizer needs a CLS token.</p>},
    {id:'tasks',title:'How the model is picked',body:<p>Each use case maps to a Hub <C>pipeline_tag</C>. The recommendation searches that tag for the most-downloaded models with known parameter counts that fit your memory. Any model ID you type in is checked against the same tag. See <D to="picking-a-base-model">Picking a base model</D>.</p>},
  ],
},
{
  slug:'preparing-a-dataset',title:'Preparing a dataset',group:'Basics',minutes:5,readFirst:'objectives',
  summary:'File formats, folders, Hub datasets, required columns and aliases per objective.',
  lead:<>ModelForge reads local files, folders and Hugging Face Hub datasets. It renames known column aliases to the names each trainer expects, then checks the result before training.</>,
  tryIt:{text:'Paste your dataset path and press Validate.',to:'/app',label:'Open in console'},
  sections:[
    {id:'sources',title:'Sources and formats',body:<>
      <ul>
        <li><b>Files:</b> <C>.parquet</C>, <C>.arrow</C>, <C>.jsonl</C>, <C>.json</C>, <C>.csv</C>, <C>.txt</C>. JSON can be a top-level array or an object holding one list of rows.</li>
        <li><b>Folders:</b> every file of the highest-priority format present is loaded together, in the order parquet, arrow, jsonl, json, csv, txt. Files in other formats in the same folder are ignored.</li>
        <li><b>Hub:</b> switch to <b>Hub</b> and enter an ID like <C>org/dataset</C>. The first split is used.</li>
      </ul>
    </>},
    {id:'columns',title:'Required columns',body:<>
      <Table head={['Use case · objective','Required columns','Accepted aliases']} mono={[]} rows={[
        ['Instruction following · supervised',<><C>prompt</C>, <C>completion</C></>,<><C>instruction</C>, <C>response</C></>],
        ['Instruction following · preference',<><C>chosen</C>, <C>rejected</C> (+ optional <C>prompt</C>)</>,<><C>preferred_response</C>, <C>dispreferred_response</C>, <C>instruction</C></>],
        ['Instruction following · reinforcement',<><C>prompt</C>, <C>answer</C></>,<><C>instruction</C>, <C>ideal_answer</C></>],
        ['Text generation · supervised',<><C>prompt</C>, <C>completion</C></>,<><C>input_text</C>, <C>generated_text</C></>],
        ['Text generation · preference',<><C>chosen</C>, <C>rejected</C> (+ optional <C>prompt</C>)</>,<><C>preferred_continuation</C>, <C>rejected_continuation</C>, <C>input_text</C></>],
        ['Text generation · reinforcement',<><C>prompt</C>, <C>answer</C></>,<><C>input_text</C>, <C>ideal_continuation</C></>],
        ['Summarization',<><C>document</C>, <C>summary</C></>,'—'],
        ['Extractive QA',<><C>context</C>, <C>question</C>, <C>answers</C></>,'—'],
      ]}/>
    </>},
    {id:'aliases',title:'How aliases work',body:<>
      <p>Aliases are renamed to the canonical columns before validation and before training, in the browser, terminal and notebook alike. A file with <C>instruction</C> and <C>response</C> trains exactly like one with <C>prompt</C> and <C>completion</C>.</p>
      <Callout tone="warn" title="Don't include both.">A dataset containing an alias and its canonical column, for example both <C>instruction</C> and <C>prompt</C>, is rejected as ambiguous. Drop or rename one of them.</Callout>
    </>},
    {id:'qa-format',title:'Extractive QA answers',body:<>
      <p><C>answers</C> holds parallel lists of answer texts and their character offsets in <C>context</C>, in the same shape as SQuAD:</p>
      <Code>{`{"context": "The library opened in 1999 on Main Street.",
 "question": "When did the library open?",
 "answers": {"text": ["1999"], "answer_start": [22]}}`}</Code>
      <p>Offsets count characters, not tokens. An off-by-one offset trains the model on the wrong span, so check them.</p>
    </>},
    {id:'validate',title:'Validating',body:<>
      <p><b>Validate</b> loads the dataset, applies the aliases, and runs the trainer's own format check for the objective you picked. It reports the normalized columns, or the first problem it finds. Common problems:</p>
      <ul>
        <li><b>Missing required column(s)</b>: rename your columns or pick the use case whose aliases match.</li>
        <li><b>Contains both an alias and its canonical column</b>: see above.</li>
        <li><b>Path not found</b>: the path is resolved on the server's machine, relative to where you ran <C>modelforge run</C>.</li>
      </ul>
      <p>GRPO can't train from streaming datasets. Use a finite, downloaded dataset.</p>
    </>},
  ],
},
];
