import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {cleanup, fireEvent, screen, waitFor} from '@testing-library/react';

beforeEach(()=>{
  document.body.innerHTML='<div id="root"></div>'; localStorage.clear();
  Object.defineProperty(window,'matchMedia',{writable:true,value:()=>({matches:false,addEventListener(){},removeEventListener(){}})});
  vi.stubGlobal('fetch',vi.fn(async (input:RequestInfo|URL)=>{const url=String(input);const data=url.endsWith('/runs')?{runs:[]}:url.endsWith('/hardware')?{gpus:[],ram_gb:16,available_disk_storage_gb:50}:url.endsWith('/system')?{use_cases:[],providers:[],name:'ModelForge',version:'4.0'}:{};return {ok:true,json:async()=>data}}));
});
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.unstubAllGlobals();vi.resetModules()});
describe('workbench routes',()=>{
  it('renders the current playground and both inference modes',async()=>{history.replaceState(null,'','/app/playground');await import('../src/main');expect(await screen.findByRole('heading',{name:'Playground'})).toBeTruthy();fireEvent.click(screen.getByRole('button',{name:'Streaming'}));expect(screen.getByRole('button',{name:'Open streaming session'})).toBeTruthy();expect(screen.getByText(/Sessions are unavailable during training/)).toBeTruthy()});
  it('retains the API token after capturing it from the startup URL',async()=>{history.replaceState(null,'','/app/settings?token=secret');await import('../src/main');await waitFor(()=>expect(localStorage.getItem('modelforge_token')).toBe('secret'));expect(location.search).toBe('');expect(await screen.findByRole('heading',{name:'Settings'})).toBeTruthy()});
});
