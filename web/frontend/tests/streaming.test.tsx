import {expect, it, vi} from 'vitest';
import {fireEvent, screen, waitFor} from '@testing-library/react';

it('opens a session, streams tokens, and stops generation', async()=>{
  document.body.innerHTML='<div id="root"></div>';
  history.replaceState(null,'','/app/playground');
  Object.defineProperty(window,'matchMedia',{writable:true,value:()=>({matches:false})});
  localStorage.setItem('modelforge_token','secret');
  const fetchMock=vi.fn(async (input:RequestInfo|URL,init?:RequestInit)=>{const url=String(input);let data:unknown={};if(url.endsWith('/runs'))data={runs:[{id:1,status:'completed',model_id:'example/model',output_dir:'/model',created_at:'2026-01-01'}]};else if(url.endsWith('/hardware'))data={gpus:[],ram_gb:16,available_disk_storage_gb:50};else if(url.endsWith('/checkpoints'))data={checkpoints:[]};else if(url.endsWith('/sessions')&&init?.method==='POST')data={session_id:'session1',note:null};return {ok:true,json:async()=>data}});
  vi.stubGlobal('fetch',fetchMock);
  class Socket {
    static OPEN=1; static latest:Socket; readyState=1;
    onopen:()=>void=()=>{}; onmessage:(event:{data:string})=>void=()=>{}; onclose:()=>void=()=>{}; onerror:()=>void=()=>{};
    send=vi.fn(); constructor(public url:string){Socket.latest=this} close(){this.onclose()}
  }
  vi.stubGlobal('WebSocket',Socket);
  await import('../src/main');
  await screen.findByRole('option',{name:/example\/model/});
  fireEvent.change(screen.getByLabelText('Run'),{target:{value:'1'}});
  fireEvent.click(screen.getByRole('button',{name:'Streaming'}));
  fireEvent.click(screen.getByRole('button',{name:'Open streaming session'}));
  await screen.findByText('Session open');
  fireEvent.change(screen.getByLabelText('Message'),{target:{value:'Hello'}});
  fireEvent.click(screen.getByRole('button',{name:'Send message'}));
  const socket=Socket.latest; socket.onopen();
  expect(socket.url).toContain('/api/playground/session1/chat?token=secret');
  expect(JSON.parse(socket.send.mock.calls[0][0] as string).messages[0].content).toBe('Hello');
  socket.onmessage({data:JSON.stringify({type:'token',text:'Hi!'})});
  await screen.findByText('Hi!');
  fireEvent.click(screen.getByRole('button',{name:'Stop generation'}));
  expect(JSON.parse(socket.send.mock.calls[1][0] as string)).toEqual({type:'stop'});
  socket.onmessage({data:JSON.stringify({type:'done'})});
  await waitFor(()=>expect(screen.getByRole('button',{name:'Close session'})).toBeTruthy());
  fireEvent.click(screen.getByRole('button',{name:'Close session'}));
  await waitFor(()=>expect(fetchMock).toHaveBeenCalledWith('/api/playground/sessions/session1',expect.objectContaining({method:'DELETE'})));
  vi.unstubAllGlobals();
});
