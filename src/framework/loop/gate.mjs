import { armPose } from "./evidence.mjs";
import { distance, moved, quaternionAngle } from "./progress.mjs";

export function poseArguments(action) {
  const args = action?.arguments;
  if (!args || typeof args.arm !== "string" || !Array.isArray(args.position) || !Array.isArray(args.quaternion)) return null;
  return args;
}

export function claimsCompletion(decision, config) {
  if (decision?.claim?.status === "complete") return true;
  if (decision?.claim?.status === "working") return false;
  const text = [decision?.thought, decision?.claim?.evidence].filter(item => typeof item === "string").join(" ");
  return new RegExp(config.claimPattern, "iu").test(text);
}

/**
 * Screen a candidate action before execution.
 * Returns { verdict: "allow" | "near_zero" | "repeat" | "hold_claim", ... }; unknown shapes are allowed.
 */
export function screenAction({ decision, state, history, config, progress }) {
  const args = poseArguments(decision?.action);
  const pose = args && armPose(state, args.arm);
  if (!args || !pose) return { verdict: claimsCompletion(decision, config) ? "hold_claim" : "allow", reason: "unscreened" };
  const translation = distance(args.position, pose.slice(0, 3));
  const rotation = quaternionAngle(args.quaternion, pose.slice(3));
  const opening = state[`${args.arm}_gripper`];
  const gripper = typeof args.opening === "number" && typeof opening === "number" ? Math.abs(args.opening - opening) : 0;
  const metrics = { translation, rotation, gripper };
  const nearZero = translation < config.nearZeroTranslation && rotation < config.nearZeroRotation && gripper < config.gripperChange;
  const last = history.at(-1);
  const previous = poseArguments(last?.action);
  const repeat = previous && previous.arm === args.arm
    && distance(previous.position, args.position) < config.repeatTranslation
    && quaternionAngle(previous.quaternion, args.quaternion) < config.repeatRotation
    && previous.opening === args.opening
    && !moved(last.observation?.state, last.nextObservation?.state, progress);
  const claim = claimsCompletion(decision, config);
  if ((nearZero || repeat) && claim) return { verdict: "hold_claim", reason: nearZero ? "idle_with_claim" : "repeat_with_claim", metrics };
  if (claim) return { verdict: "hold_claim", reason: "explicit_claim", metrics, passThrough: true };
  if (nearZero) return { verdict: "near_zero", reason: `target within ${translation.toFixed(4)} m and ${rotation.toFixed(3)} rad of the current pose with no gripper change`, metrics };
  if (repeat) return { verdict: "repeat", reason: "same command as the previous step, which did not move the robot", metrics };
  return { verdict: "allow", metrics };
}
