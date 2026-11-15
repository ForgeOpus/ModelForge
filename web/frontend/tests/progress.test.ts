import {describe, expect, it} from 'vitest';
import {deriveProgress, normalizeTrainingEvent, type TrainingEvent} from '../src/lib/progress';
const log=(step:number,ts?:number):TrainingEvent=>({run_id:1,type:'log',ts,payload:{step,max_steps:2000,logs:{loss:1.5}}});
describe('training progress',()=>{
  it('calculates ETA using Unix seconds',()=>{const result=deriveProgress([log(0,1700000000),log(1000,1700001000)]);expect(result.etaSeconds).toBe(1000);expect(result.latest?.loss).toBe(1.5)});
  it('normalizes missing timestamps and avoids invented rates',()=>{expect(normalizeTrainingEvent(log(0),1700000000000).ts).toBe(1700000000);expect(deriveProgress([log(0),log(100)],1700000000000).etaSeconds).toBeNull()});
  it('clamps an overrun and clears ETA on completion',()=>{expect(deriveProgress([log(0,1),log(3000,11)]).etaSeconds).toBe(0);expect(deriveProgress([log(0,1),log(100,11),{run_id:1,type:'completed'}]).etaSeconds).toBeNull()});
});
