import React, {useState} from 'react';
import {Link} from 'react-router-dom';
import {AlertTriangle, Info, Lightbulb} from 'lucide-react';

export type Group='Get started'|'Basics'|'Intermediate'|'Advanced'|'Reference';
export const GROUPS:Group[]=['Get started','Basics','Intermediate','Advanced','Reference'];
export const LEVELS:Partial<Record<Group,number>>={Basics:1,Intermediate:2,Advanced:3};

export interface Section{id:string;title:string;body:React.ReactNode}
export interface DocPage{
  slug:string;
  title:string;
  group:Group;
  minutes:number;
  lead:React.ReactNode;
  /** Plain-text summary used by search. */
  summary:string;
  readFirst?:string;
  sections:Section[];
  tryIt?:{text:string;to:string;label:string};
}

export const docPath=(slug:string)=>'/app/docs/'+slug;

/** Link to another docs page. */
export function D({to,children}:{to:string;children:React.ReactNode}){return <Link to={docPath(to)}>{children}</Link>}

export function C({children}:{children:React.ReactNode}){return <code>{children}</code>}

export function Callout({tone='accent',title,children}:{tone?:'accent'|'warn'|'info';title?:string;children:React.ReactNode}){
  const Icon=tone==='warn'?AlertTriangle:tone==='info'?Info:Lightbulb;
  return <div className={'mfd-callout '+tone}><Icon size={15}/><span>{title&&<b>{title} </b>}{children}</span></div>;
}

/** Code block. Lines starting with "# " render as comments; "$ " renders a prompt. */
export function Code({children,label}:{children:string;label?:string}){
  const [copied,setCopied]=useState(false);
  const text=children.replace(/^\n/,'').replace(/\s+$/,'');
  const plain=text.split('\n').map(l=>l.replace(/^\$ /,'')).join('\n');
  return <div className="mfd-code">
    <pre>{label&&<span className="label">{label}</span>}{text.split('\n').map((line,i)=><React.Fragment key={i}>{i>0&&'\n'}{line.startsWith('#')?<span className="c">{line}</span>:line.startsWith('$ ')?<><span className="p">$ </span>{line.slice(2)}</>:line}</React.Fragment>)}</pre>
    <button type="button" className="mfd-copy" onClick={()=>{void navigator.clipboard?.writeText(plain).then(()=>{setCopied(true);setTimeout(()=>setCopied(false),1200)})}}>{copied?'Copied':'Copy'}</button>
  </div>;
}

export function Table({head,rows,mono=[0],right=[]}:{head:string[];rows:React.ReactNode[][];mono?:number[];right?:number[]}){
  return <div className="mfd-tbl"><table><thead><tr>{head.map((h,i)=><th key={i} className={right.includes(i)?'r':''}>{h}</th>)}</tr></thead><tbody>{rows.map((r,i)=><tr key={i}>{r.map((c,j)=><td key={j} className={[mono.includes(j)?'mono':'',right.includes(j)?'r':''].join(' ').trim()||undefined}>{c}</td>)}</tr>)}</tbody></table></div>;
}

export function Steps({items}:{items:[string,React.ReactNode][]}){
  return <ol className="mfd-steps">{items.map(([t,b])=><li key={t}><b>{t}</b>{b}</li>)}</ol>;
}

export function Cards({items}:{items:{to:string;title:string;text:string}[]}){
  return <div className="mfd-cards">{items.map(c=><Link key={c.to+c.title} to={docPath(c.to)}><b>{c.title}</b><span>{c.text}</span></Link>)}</div>;
}

export function Figure({children,eq,caption,note}:{children:React.ReactNode;eq?:React.ReactNode;caption?:string;note?:string}){
  return <figure className="mfd-diagram" style={{margin:'18px 0 0'}}>{children}{eq&&<div className="mfd-eq">{eq}</div>}{(caption||note)&&<figcaption className="mfd-cap"><span>{caption}</span>{note&&<span className="mono">{note}</span>}</figcaption>}</figure>;
}
