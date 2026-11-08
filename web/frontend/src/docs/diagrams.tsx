import {Figure} from './ui';

/* Diagrams use classes from docs.css (.d-*) so they follow light and dark themes. */

function Defs(){return <defs><marker id="mfd-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10z" className="d-ahead"/></marker></defs>}
const Arrow=({d}:{d:string})=><path d={d} className="d-arrow" markerEnd="url(#mfd-ah)"/>;

export function LoraDiagram(){
  return <Figure eq={<>h = W·x + <span className="a">(α / r) · B·A·x</span></>} caption="Grey is frozen. Orange is trained." note="2048 × 2048, r = 16 → 65.5K trainable of 4.19M (1.6%)">
    <svg viewBox="0 0 640 190" role="img" aria-label="LoRA adds a trained low-rank path B·A beside a frozen weight matrix W">
      <Defs/>
      <text x="30" y="99" className="d-t ink" fontSize="15" textAnchor="middle">x</text>
      <Arrow d="M44 95 L120 55"/><Arrow d="M44 95 L120 140"/>
      <rect x="124" y="10" width="150" height="90" rx="6" className="d-frozen"/>
      <text x="199" y="52" className="d-t" fontSize="18" textAnchor="middle">W</text>
      <text x="199" y="72" className="d-t dim" fontSize="11" textAnchor="middle">frozen · d × d</text>
      <rect x="124" y="122" width="96" height="16" rx="3" className="d-train"/>
      <text x="172" y="134" className="d-t acc" fontSize="10.5" textAnchor="middle">A · r × d</text>
      <Arrow d="M222 130 L250 130"/>
      <rect x="254" y="112" width="16" height="64" rx="3" className="d-train"/>
      <text x="292" y="148" className="d-t acc" fontSize="12" textAnchor="middle">B</text>
      <text x="278" y="162" className="d-t acc" fontSize="10.5">d × r</text>
      <Arrow d="M280 55 L380 88"/><Arrow d="M330 140 L380 108"/>
      <circle cx="396" cy="98" r="14" className="d-ring"/>
      <text x="396" y="103" className="d-t ink" fontSize="16" textAnchor="middle">+</text>
      <Arrow d="M412 98 L480 98"/>
      <text x="500" y="103" className="d-t ink" fontSize="15" textAnchor="middle">h</text>
      <text x="530" y="60" className="d-t dim" fontSize="11">trained:</text>
      <text x="530" y="78" className="d-t acc" fontSize="11">A and B only</text>
      <text x="530" y="126" className="d-t dim" fontSize="11">frozen:</text>
      <text x="530" y="144" className="d-t" fontSize="11">W (bf16, 8-bit</text>
      <text x="530" y="160" className="d-t" fontSize="11">or 4-bit)</text>
    </svg>
  </Figure>;
}

/** Stacked bars: GB needed to train a 7B model at each tier (weights + adapter + activations). */
export function VramBars(){
  const rows=[
    {name:'Full fine-tune',w:126,ad:0},
    {name:'LoRA bf16',w:14,ad:0.1},
    {name:'QLoRA 8-bit',w:7,ad:0.1},
    {name:'QLoRA 4-bit',w:3.5,ad:0.1},
  ];
  const max=130,x0=120,width=420,sc=width/max;
  return <Figure caption="7B parameters. Grey is weights, plus gradients and optimizer state for full fine-tuning. Orange is the 2 GB activation budget." note="bars to scale, 0–130 GB">
    <svg viewBox="0 0 640 200" role="img" aria-label="Memory needed to train a 7B model at each tier">
      {rows.map((r,i)=>{const y=14+i*44;const total=r.w+r.ad+2;return <g key={r.name}>
        <text x="0" y={y+17} className="d-t" fontSize="12">{r.name}</text>
        <rect x={x0} y={y} width={Math.max(r.w*sc,1)} height="26" rx="3" className="d-frozen"/>
        <rect x={x0+r.w*sc} y={y} width={(r.ad+2)*sc} height="26" rx="2" className="d-train"/>
        <text x={x0+(r.w+r.ad+2)*sc+8} y={y+17} className="d-t ink" fontSize="12">{total.toFixed(1)} GB</text>
      </g>})}
      {[24,48,80].map(g=><g key={g}><line x1={x0+g*sc} x2={x0+g*sc} y1="6" y2="186" className="d-grid"/><text x={x0+g*sc} y="198" className="d-t dim" fontSize="10" textAnchor="middle">{g} GB</text></g>)}
    </svg>
  </Figure>;
}

/** Data parallel keeps a whole copy per GPU; FSDP shards the copy across GPUs. */
export function ParallelDiagram(){
  const gpu=(x:number,label:string,shard:boolean,k:number)=><g key={label+x}>
    <rect x={x} y="40" width="120" height="96" rx="8" className="d-box"/>
    <text x={x+60} y="30" className="d-t dim" fontSize="11" textAnchor="middle">{label}</text>
    {shard?[0,1,2,3].map(j=><rect key={j} x={x+12} y={52+j*19} width="96" height="15" rx="2" className={j%2===k?'d-train':'d-ghost'}/>)
      :[0,1,2,3].map(j=><rect key={j} x={x+12} y={52+j*19} width="96" height="15" rx="2" className="d-frozen"/>)}
  </g>;
  return <Figure caption="Left: each GPU holds the full model and sees different batches. Right: each GPU holds half of every layer group and gathers the rest when needed." note="2 GPUs">
    <svg viewBox="0 0 640 160" role="img" aria-label="Data parallel compared with FSDP across two GPUs">
      <text x="10" y="14" className="d-t ink" fontSize="12">Data parallel</text>
      {gpu(10,'GPU 0 · full copy',false,0)}{gpu(150,'GPU 1 · full copy',false,0)}
      <line x1="320" x2="320" y1="10" y2="150" className="d-grid"/>
      <text x="340" y="14" className="d-t ink" fontSize="12">FSDP (full_shard)</text>
      {gpu(340,'GPU 0 · shards 1, 3',true,0)}{gpu(480,'GPU 1 · shards 2, 4',true,1)}
    </svg>
  </Figure>;
}

/** Packing fills each training row with several examples instead of padding. */
export function PackingDiagram(){
  const ex=[180,90,260,120,140];
  const row=(y:number,parts:number[])=>{let x=110;const out=parts.map((w,i)=>{const r=<rect key={i} x={x} y={y} width={w-3} height="18" rx="3" className={i%2?'d-train':'d-train alt'}/>;x+=w;return r});out.push(<rect key="pad" x={x} y={y} width={Math.max(0,520-x)} height="18" rx="3" className="d-ghost"/>);return out};
  return <Figure caption="The same five examples. Orange blocks are examples, grey is padding. Packing needs two rows instead of five." note="row width = max_length">
    <svg viewBox="0 0 640 182" role="img" aria-label="Unpacked rows padded to max length compared with packed rows">
      <text x="0" y="22" className="d-t dim" fontSize="11">unpacked</text>
      {ex.map((w,i)=><g key={i}>{row(8+i*22,[w])}</g>)}
      <text x="0" y="154" className="d-t dim" fontSize="11">packed</text>
      {[[180,90,120],[260,140]].map((p,i)=><g key={i}>{row(140+i*22,p)}</g>)}
    </svg>
  </Figure>;
}

/** Tier decision flow. */
export function TierFlow(){
  const box=(x:number,y:number,w:number,t:string,s:string,cls='d-box')=><g><rect x={x} y={y} width={w} height="44" rx="7" className={cls}/><text x={x+w/2} y={y+19} className="d-t ink" fontSize="12" textAnchor="middle">{t}</text><text x={x+w/2} y={y+34} className="d-t dim" fontSize="10.5" textAnchor="middle">{s}</text></g>;
  return <Figure caption="The first tier whose parameter ceiling is above zero wins." note="order: full → LoRA bf16 → 8-bit → 4-bit">
    <svg viewBox="0 0 640 170" role="img" aria-label="How ModelForge picks a precision tier">
      <Defs/>
      {box(0,10,150,'Detect hardware','sum VRAM across GPUs')}
      <Arrow d="M152 32 L186 32"/>
      {box(190,10,170,'Adjust for strategy','extra copies, batch size')}
      <Arrow d="M362 32 L396 32"/>
      {box(400,10,240,'Ceiling per tier','P = (VRAM − overhead) / bytes')}
      <Arrow d="M520 56 L520 92"/>
      <Arrow d="M75 56 L75 96"/><Arrow d="M275 56 L275 96"/>
      {box(0,100,175,'CPU only','qlora_4bit, ≤ 500M params','d-ghostbox')}
      {box(195,100,165,'Apple MPS','full, else LoRA bf16','d-ghostbox')}
      {box(400,100,240,'CUDA','first ceiling > 0','d-trainbox')}
    </svg>
  </Figure>;
}
