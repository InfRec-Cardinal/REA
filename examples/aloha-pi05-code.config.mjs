import alohaReal from "./aloha-real.config.mjs";

// Classic ROS1 ALOHA with pi0.5 and Python Code as the two candidates.
export default function ({ env = process.env } = {}) {
  const config = alohaReal({ env: { ...env, ALOHA_POLICY: "pi05-code" } });
  config.name = "aloha-pi05-code";
  config.episodes[0].id = config.name;
  return config;
}
