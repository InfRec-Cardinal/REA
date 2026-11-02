import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import test from "node:test";
import { actionErrors, evaluateOffline, policyObservation } from "../scripts/behavior_offline.mjs";

function sample(id = "sample") {
  return {
    id, taskName: "turning_on_radio", episodeIndex: 0, frameIndex: 100,
    referenceAction: Array(23).fill(0.25), reward: "hidden-reward", source: { future: "hidden-future" },
    observation: {
      text: "Turn on the radio.", state: Array(61).fill(0),
      images: [{ name: "head", mimeType: "image/jpeg", data: "fixture" }],
      extra: "hidden-extra",
    },
  };
}

test("offline policy receives only present sensor observations and task instruction", () => {
  const input = sample();
  const observation = policyObservation(input);
  assert.equal(observation.text, input.observation.text);
  assert.deepEqual(observation.images, input.observation.images);
  assert.equal(observation.actions[0].name, "joint_action");
  assert.equal(observation.referenceAction, undefined);
  assert.equal(observation.frameIndex, undefined);
  assert.doesNotMatch(JSON.stringify(observation), /hidden-|referenceAction/u);
});

test("native action error groups keep base, joints and grippers distinct", () => {
  const reference = Array(23).fill(0);
  const values = Array(23).fill(0);
  values[0] = 0.6;
  values[7] = 0.7;
  values[14] = 1;
  const errors = actionErrors(values, reference);
  assert.ok(Math.abs(errors.baseCommand - 0.2) < 1e-12);
  assert.ok(Math.abs(errors.leftArm - 0.1) < 1e-12);
  assert.equal(errors.leftGripper, 1);
  assert.equal(errors.rightArm, 0);
});

test("invalid policy calls remain errors; comparison and hold use the same samples", async () => {
  const outputDir = await mkdtemp(join(tmpdir(), "behavior-offline-"));
  let calls = 0;
  let closed = false;
  try {
    const first = sample("first");
    const second = sample("second");
    second.referenceAction = Array(23).fill(0.75);
    const policy = {
      id: "test-policy",
      async decide(input) {
        assert.deepEqual(input.history, []);
        assert.equal(input.notes, "");
        assert.equal(input.referenceAction, undefined);
        if (++calls === 2) return { action: { name: "joint_action", arguments: { values: [0] } }, usage: { total_tokens: 8 } };
        return { action: { name: "joint_action", arguments: { values: first.referenceAction } }, usage: { total_tokens: 12 } };
      },
      async close() { closed = true; },
    };
    const result = await evaluateOffline({ samples: [first, second], policy, outputDir });
    assert.equal(result.samples, 2);
    assert.equal(result.validActions, 1);
    assert.equal(result.errors, 1);
    assert.equal(result.perGroupMAE.baseCommand.agent, 0);
    assert.equal(result.perGroupMAE.baseCommand.holdOnSameSamples, 0.25);
    assert.equal(result.simulationSteps, 0);
    assert.equal(result.taskSuccessRate, null);
    assert.equal(result.qScore, null);
    assert.equal(result.usageComplete, true);
    assert.equal(result.usage.total_tokens, 20);
    assert.equal(closed, true);
    const records = (await readFile(join(outputDir, "decisions.jsonl"), "utf8")).trim().split("\n").map(JSON.parse);
    assert.equal(records[1].error.code, "invalid_action");
    await assert.rejects(evaluateOffline({ samples: [first], policy, outputDir }), { code: "EEXIST" });
    assert.equal((await readFile(join(outputDir, "decisions.jsonl"), "utf8")).trim().split("\n").length, 2);
  } finally { await rm(outputDir, { recursive: true, force: true }); }
});
