import {useEffect, useMemo, useRef, useState} from 'react';
import {Link, useNavigate, useParams} from 'react-router-dom';
import {Menu, Moon, Search, Sun, X} from 'lucide-react';
import './docs.css';
import {PAGES} from './pages';
import {GROUPS, LEVELS, docPath} from './ui';

const bySlug=new Map(PAGES.map(p=>[p.slug,p]));
const ORDER=GROUPS.flatMap(g=>PAGES.filter(p=>p.group===g));

function Bars({n}:{n:number}){return <span className="mfd-lvl" aria-hidden="true">{[1,2,3].map(i=><i key={i} className={i<=n?'f':''}/>)}</span>}

function useDark(){
  const read=()=>document.documentElement.dataset.theme==='dark'||(!document.documentElement.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches);
  const [dark,setDark]=useState(read);
  const toggle=()=>{const next=!dark;setDark(next);document.documentElement.dataset.theme=next?'dark':'light';localStorage.setItem('modelforge_theme',next?'dark':'light')};
  return [dark,toggle] as const;
}

function DocsSearch(){
  const [q,setQ]=useState('');
  const [open,setOpen]=useState(false);
  const [sel,setSel]=useState(0);
  const input=useRef<HTMLInputElement>(null);
  const navigate=useNavigate();
  useEffect(()=>{const onKey=(e:KeyboardEvent)=>{if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();input.current?.focus();setOpen(true)}};addEventListener('keydown',onKey);return ()=>removeEventListener('keydown',onKey)},[]);
  const results=useMemo(()=>{const terms=q.toLowerCase().split(/\s+/).filter(Boolean);if(!terms.length)return [];return ORDER.map(p=>{const title=p.title.toLowerCase();const hay=(p.title+' '+p.summary+' '+p.sections.map(s=>s.title).join(' ')).toLowerCase();if(!terms.every(t=>hay.includes(t)))return null;return {p,score:terms.filter(t=>title.includes(t)).length}}).filter((x):x is {p:typeof PAGES[number];score:number}=>!!x).sort((a,b)=>b.score-a.score).slice(0,8).map(x=>x.p)},[q]);
  const go=(slug:string)=>{navigate(docPath(slug));setQ('');setOpen(false);input.current?.blur()};
  return <div className="mfd-search">
    <Search size={14}/>
    <input ref={input} value={q} placeholder="Search docs…" aria-label="Search docs" onFocus={()=>setOpen(true)} onBlur={()=>setTimeout(()=>setOpen(false),120)} onChange={e=>{setQ(e.target.value);setSel(0)}} onKeyDown={e=>{if(e.key==='ArrowDown'){e.preventDefault();setSel(s=>Math.min(s+1,results.length-1))}else if(e.key==='ArrowUp'){e.preventDefault();setSel(s=>Math.max(s-1,0))}else if(e.key==='Enter'&&results[sel])go(results[sel].slug);else if(e.key==='Escape'){setQ('');input.current?.blur()}}}/>
    <kbd>⌘K</kbd>
    {open&&q.trim()&&<div className="mfd-results" role="listbox">{results.length?results.map((p,i)=><a key={p.slug} href={docPath(p.slug)} className={i===sel?'on':''} onMouseDown={e=>{e.preventDefault();go(p.slug)}}>{p.title}<span>{p.group} · {p.summary}</span></a>):<p>No pages match “{q}”.</p>}</div>}
  </div>;
}

export function DocsSite(){
  const {slug}=useParams();
  // Unknown or missing slugs show the introduction rather than a blank page.
  const page=(slug&&bySlug.get(slug))||PAGES[0];
  const [dark,toggleDark]=useDark();
  const [menu,setMenu]=useState(false);
  const [active,setActive]=useState('');

  useEffect(()=>{document.title=page.title+' — ModelForge Docs';setMenu(false);setActive(page.sections[0]?.id||'');if(!location.hash)window.scrollTo(0,0);else document.getElementById(location.hash.slice(1))?.scrollIntoView()},[page]);

  // Highlight the section nearest the top of the viewport in "On this page".
  useEffect(()=>{
    const heads=page.sections.map(s=>document.getElementById(s.id)).filter((h):h is HTMLElement=>!!h);
    const onScroll=()=>{let current=heads[0]?.id||'';for(const h of heads){if(h.getBoundingClientRect().top<120)current=h.id}if(scrollY>0&&innerHeight+scrollY>=document.body.scrollHeight-4)current=heads[heads.length-1]?.id||current;setActive(current)};
    onScroll();addEventListener('scroll',onScroll,{passive:true});return ()=>removeEventListener('scroll',onScroll);
  },[page]);

  const i=ORDER.indexOf(page);
  const prev=ORDER[i-1],next=ORDER[i+1];
  const readFirst=page.readFirst?bySlug.get(page.readFirst):undefined;
  const level=LEVELS[page.group];
  const top=page.group==='Reference'?(page.slug==='cli-commands'?'CLI':'Reference'):'Guides';

  return <div className="mfd">
    <header className="mfd-nav">
      <button className="mfd-icon mfd-menu" aria-label={menu?'Close navigation':'Open navigation'} onClick={()=>setMenu(!menu)}>{menu?<X size={16}/>:<Menu size={16}/>}</button>
      <Link className="mfd-brand" to="/"><img src="/logo.svg" alt=""/><b>ModelForge</b><span className="tag">Docs</span></Link>
      <DocsSearch/>
      <div style={{flex:1}}/>
      <nav className="mfd-links" aria-label="Docs sections">
        <Link className={top==='Guides'?'on':''} to={docPath('introduction')}>Guides</Link>
        <Link className={top==='Reference'?'on':''} to={docPath('recommendation-fields')}>Reference</Link>
        <Link className={top==='CLI'?'on':''} to={docPath('cli-commands')}>CLI</Link>
        <a href="https://github.com/ForgeOpus/ModelForge" target="_blank" rel="noreferrer">GitHub</a>
      </nav>
      <button className="mfd-icon" aria-label="Toggle theme" onClick={toggleDark}>{dark?<Sun size={15}/>:<Moon size={15}/>}</button>
      <Link className="mfd-btn primary" to="/app">Open console</Link>
    </header>

    <div className="mfd-grid">
      <aside className={'mfd-side'+(menu?' open':'')} aria-label="Documentation">
        {GROUPS.map(g=><div className="mfd-group" key={g}>
          <div className="mfd-gl">{LEVELS[g]&&<Bars n={LEVELS[g]!}/>}{g}</div>
          {PAGES.filter(p=>p.group===g).map(p=><Link key={p.slug} to={docPath(p.slug)} className={p===page?'on':''} aria-current={p===page?'page':undefined}>{p.title}</Link>)}
        </div>)}
      </aside>

      <article className="mfd-article">
        <div className="mfd-bc">{page.group==='Reference'?<span>Reference</span>:<><span>Guides</span><span className="sep">/</span><span>{page.group}</span></>}<span className="sep">/</span><b>{page.title}</b></div>
        <h1>{page.title}</h1>
        <div className="mfd-meta">
          {level?<span className="mfd-chip">{page.group}</span>:<span className="mfd-chip neutral">{page.group}</span>}
          <span>{page.minutes} min read</span>
          {readFirst&&<><span>·</span><span>Read first: <Link to={docPath(readFirst.slug)}>{readFirst.title}</Link></span></>}
        </div>
        <p className="mfd-lead">{page.lead}</p>
        {page.sections.map(s=><section key={s.id}><h2 id={s.id}>{s.title}</h2>{s.body}</section>)}
        <nav className="mfd-pn" aria-label="Pagination">
          {prev&&<Link to={docPath(prev.slug)}><span>Previous</span><b>← {prev.title}</b></Link>}
          {next&&<Link className="next" to={docPath(next.slug)}><span>Next</span><b>{next.title} →</b></Link>}
        </nav>
      </article>

      <aside className="mfd-toc" aria-label="On this page">
        <div className="tl">On this page</div>
        {page.sections.map(s=><a key={s.id} href={'#'+s.id} className={active===s.id?'on':''} onClick={e=>{e.preventDefault();document.getElementById(s.id)?.scrollIntoView({behavior:'smooth'});history.replaceState(null,'','#'+s.id)}}>{s.title}</a>)}
        {page.tryIt&&<div className="mfd-try"><b>Try it</b><p>{page.tryIt.text}</p><Link className="mfd-btn" to={page.tryIt.to}>{page.tryIt.label}</Link></div>}
      </aside>
    </div>
  </div>;
}

