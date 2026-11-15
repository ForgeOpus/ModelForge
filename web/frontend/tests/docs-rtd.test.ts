import {describe, expect, it} from 'vitest';
import {PAGES} from '../src/docs/pages';
import {RTD_PAGES, rtdUrl} from '../src/docs/rtd';

describe('Read the Docs links', ()=>{
  it('maps every in-app docs page to a Read the Docs page', ()=>{
    const missing=PAGES.map(p=>p.slug).filter(s=>!(s in RTD_PAGES));
    expect(missing).toEqual([]);
  });
  it('has no stale mappings', ()=>{
    const slugs=new Set(PAGES.map(p=>p.slug));
    expect(Object.keys(RTD_PAGES).filter(s=>!slugs.has(s))).toEqual([]);
  });
  it('builds absolute URLs', ()=>{
    expect(rtdUrl('http-api')).toBe('https://modelforge-finetuning.readthedocs.io/en/latest/api-reference/rest-api/');
  });
});
