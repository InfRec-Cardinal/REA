import alohaReal from "./aloha-real.config.mjs";

// Classic ROS1 ALOHA with pi0.5, LingBot-VLA and Python Code together.
export default function ({ env = process.env } = {}) {
  const config = alohaReal({ env: { ...env, ALOHA_POLICY: "hybrid" } });
  config.name = "aloha-pi05-lingbot-code";
  config.episodes[0].id = config.name;
  return config;
}
