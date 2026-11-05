import {useEffect, useRef, useState} from 'react';
import markup from './landing-reference.html?raw';
import './landing-reference.css';

const tiers = [
  {name:'Full fine-tune', bpp:18, adapter:0},
  {name:'LoRA bf16', bpp:2, adapter:0.1},
  {name:'QLoRA 8-bit', bpp:1, adapter:0.1},
  {name:'QLoRA 4-bit', bpp:0.5, adapter:0.1},
];
const sizes = [8,12,16,24,48,80];
const format = (b:number) => b>=10 ? b.toFixed(1)+'B' : b>=1 ? b.toFixed(2)+'B' : Math.round(b*1000)+'M';
function currentTheme(): 'light'|'dark' {
  const saved = localStorage.getItem('modelforge_theme');
  return saved==='light'||saved==='dark' ? saved : matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';
}
export function LandingReference(){
  const root = useRef<HTMLDivElement>(null);
  const [theme,setTheme] = useState(currentTheme);
  useEffect(()=>{
    const element=root.current;
    if(!element)return;
    let vram=24;
    const chips=element.querySelector<HTMLElement>('#vram')!;
    const body=element.querySelector<HTMLElement>('#tiers')!;
    const verdict=element.querySelector<HTMLElement>('#verdict')!;
    const render=()=>{
      chips.innerHTML=sizes.map(v=>`<button type="button" class="${v===vram?'on':''}" data-v="${v}" aria-pressed="${v===vram}">${v} GB</button>`).join('');
      const pm=tiers.map(t=>Math.max(0,(vram-t.adapter-2)/t.bpp));
      const chosen=pm.findIndex(p=>p>0);
      const logMax=Math.log10(160),logMin=Math.log10(0.1);
      body.innerHTML=tiers.map((t,i)=>{const width=pm[i]>0?((Math.log10(pm[i])-logMin)/(logMax-logMin))*100:0;return `<div class="tier ${i===chosen?'on':''}"><span class="nm">${t.name}<small>${t.bpp} B / param</small></span><div class="tr"><i style="width:${width.toFixed(1)}%"></i></div><span class="val">${format(pm[i])}</span></div>`}).join('');
      verdict.innerHTML=`On a <b>${vram} GB</b> card ModelForge fully fine-tunes models up to <b>${format(pm[0])}</b> parameters. Choose LoRA bf16 to reach <b>${format(pm[1])}</b>, or QLoRA 4-bit for up to <b>${format(pm[3])}</b>.`;
    };
    const onClick=(event:MouseEvent)=>{
      const target=event.target as Element;
      const chip=target.closest<HTMLButtonElement>('#vram button');
      if(chip){vram=Number(chip.dataset.v);render();return}
      if(target.closest('[data-theme-toggle]')){
        const next=currentTheme()==='dark'?'light':'dark';
        localStorage.setItem('modelforge_theme',next);document.documentElement.dataset.theme=next;setTheme(next);return;
      }
      const copy=target.closest<HTMLButtonElement>('.install-head .copy');
      if(copy){void navigator.clipboard.writeText('pip install modelforge-finetuning\nmodelforge run').then(()=>{copy.setAttribute('aria-label','Install commands copied');}).catch(()=>{copy.setAttribute('aria-label','Copy unavailable');});}
    };
    const scale=()=>{
      const hero=element.querySelector<HTMLElement>('#heroShot')!;
      const heroFrame=hero.querySelector<HTMLIFrameElement>('iframe')!;
      const s=hero.clientWidth/1440;hero.style.setProperty('--s',String(s));heroFrame.style.transform=`scale(${s})`;
      const run=element.querySelector<HTMLElement>('#runShot')!;
      const runFrame=run.querySelector<HTMLIFrameElement>('iframe')!;
      const rs=run.clientWidth/1180;runFrame.style.transform=`scale(${rs})`;runFrame.style.left=`${-236*rs}px`;runFrame.style.top=`${-52*rs}px`;
    };
    render();element.addEventListener('click',onClick);window.addEventListener('resize',scale);
    const observer=new ResizeObserver(scale);observer.observe(element);scale();
    return()=>{element.removeEventListener('click',onClick);window.removeEventListener('resize',scale);observer.disconnect()};
  },[]);
  useEffect(()=>{
    root.current?.querySelectorAll<HTMLIFrameElement>('iframe[data-src]').forEach(frame=>{frame.src=`${frame.dataset.src}?theme=${theme}`});
  },[theme]);
  return <div ref={root} className="landing-reference" data-theme={theme} dangerouslySetInnerHTML={{__html:markup}}/>;
}
