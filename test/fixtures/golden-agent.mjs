import { createAgent } from "../../src/agent.mjs";
import { loadConfig } from "../../src/config.mjs";
import { createSessionStore } from "../../src/sessions.mjs";

export function goldenAgent() {
  const engine = name => Object.assign(async () => ({ token: "X", tokenReadyS: 0 }), { engine: name });
  const config = loadConfig({ REA_API_KEY: "golden", REA_MODEL: "golden-model", G1_USE_LIDAR: "0" });
  return createAgent({
    camera: { capture: async () => ({ imageB64: "aGk=", depth: { center_m: 2, left_m: 2, right_m: 2 } }) },
    engines: { prime: engine("prime"), token: engine("token"), tool: engine("tool"), react: engine("react") },
    executor: { actuatorInfo: () => ({ serialized: true, latestWins: false }), armState: () => ({ held: null, holding_tokens: ["U", "P"] }) },
    config,
    sessions: createSessionStore(),
    logger: { info() {}, warn() {}, error() {} },
  });
}
