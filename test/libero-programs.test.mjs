import test from 'node:test';
import assert from 'node:assert/strict';
import { createLiberoProgramEnvironment } from '../src/environments/libero-programs.mjs';

test('program adapter exposes physical feedback and reads success only in final score', async () => {
  const requests = [];
  const observed = { observationId: '1:1', frame: { capturedAt: new Date().toISOString(), mimeType: 'image/jpeg', imageB64: '' },
    task: { language: 'open the bottom drawer' }, proprio: { gripper: 'open' }, steps: 100, stepsLeft: 3900 };
  let resetOptions;
  const client = { async request(command, options) {
    requests.push(command);
    if (command === 'reset') resetOptions = options;
    if (command === 'program_chunk') return { candidateId: 'sdk', programDone: true, exception: null, truncated: false, steps: 100 };
    if (command === 'score') return { task_completed: true, steps: 100 };
    return observed;
  } };
  const environment = createLiberoProgramEnvironment({ client, candidateIds: ['sdk'], maxDurationMs: 900000 });
  const reset = await environment.reset({ task: { suite: 'libero_goal_task', task: 'middle_drawer' }, seed: 1 });
  assert.equal(reset.observation.text, 'open the bottom drawer');
  assert.equal(resetOptions.wall_budget_s, 900);
  const result = await environment.step({ name: 'continue_program', arguments: { candidateId: 'sdk' } });
  assert.equal(result.terminated, true);
  assert.equal(result.success, undefined);
  assert.equal(requests.includes('score'), false);
  await environment.stop();
  assert.equal(result.info.capability.status, 'returned');
  assert.equal(environment.resumable, true);
  const score = await environment.score();
  assert.equal(score.task_completed, true);
  assert.equal(score.success, true);
  assert.equal(requests.filter(command => command === 'score').length, 1);
});
