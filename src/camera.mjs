import { execFile } from "node:child_process";

export class CameraError extends Error {
  constructor(message, cause) {
    super(message, cause ? { cause } : undefined);
    this.name = "CameraError";
    this.errorType = "capture_error";
    this.retryable = false;
  }
}

function normaliseFrame(payload) {
  return {
    imageB64: payload.image_b64,
    mimeType: "image/jpeg",
    depth: payload.depth ?? null,
    scene: payload.scene && typeof payload.scene === "object" ? payload.scene : null,
    capturedAt: payload.captured_at ?? null,
    width: payload.width ?? null,
    height: payload.height ?? null,
    jpegBytes: payload.jpeg_bytes ?? null,
    timing: payload.timing ?? null,
  };
}

/**
 * Persistent capture through the sense worker (default).
 * numpy/cv2 stay loaded and the depth scale is cached, so a capture is the socket round-trip plus
 * the JPEG encode: ~18 ms against ~1.1-1.7 s for the subprocess path below.
 */
export function createWorkerCamera({ worker, timeoutMs = 5000 }) {
  return {
    kind: "worker",
    async capture(cfg = {}) {
      // depthOnly skips the JPEG encode: the gate's live re-read needs the depth summary, not pixels.
      const depthOnly = cfg.depthOnly === true;
      const reply = await worker.request(
        {
          op: "capture",
          max_width: cfg.maxWidth ?? 320,
          quality: cfg.quality ?? 40,
          filter: cfg.filter ?? "",
          no_depth: !depthOnly && cfg.includeDepth === false && cfg.needDepth === false,
          depth_only: depthOnly,
          // the gate's live re-read keeps only the depth summary; a scene hash there is wasted work
          no_scene: depthOnly,
        },
        timeoutMs,
      );
      if (!reply?.ok) throw new CameraError(reply?.error ?? "capture failed");
      if (depthOnly) {
        if (!reply.depth) throw new CameraError("sense worker returned no depth summary");
        return normaliseFrame(reply);
      }
      if (typeof reply.image_b64 !== "string" || reply.image_b64.length === 0) {
        throw new CameraError("sense worker returned no JPEG image");
      }
      return normaliseFrame(reply);
    },
  };
}

function executeFile(file, args, options) {
  return new Promise((resolve, reject) => {
    execFile(file, args, options, (error, stdout, stderr) => {
      if (error) {
        error.stdout = stdout;
        error.stderr = stderr;
        reject(error);
      } else {
        resolve({ stdout, stderr });
      }
    });
  });
}

function lastJson(text) {
  const lines = String(text ?? "")
    .split(/\r?\n/u)
    .filter((line) => line.trim());
  if (lines.length === 0) return null;
  try {
    return JSON.parse(lines.at(-1));
  } catch {
    return null;
  }
}

/**
 * One-shot capture by spawning g1_u2_sense.py. Kept as a fallback for hosts
 * where the persistent worker cannot run; correct but roughly 60x slower per frame.
 */
export function createSubprocessCamera({
  python,
  script,
  socketPath,
  timeoutMs = 5000,
  maxWidth = 640,
  jpegQuality = 80,
  execFileFn = executeFile,
  env = process.env,
}) {
  return {
    kind: "subprocess",
    async capture(cfg = {}, { signal } = {}) {
      const args = [
        script,
        "--socket",
        socketPath,
        "--timeout",
        String(timeoutMs / 1000),
        "snapshot",
        "--max-width",
        String(cfg.maxWidth ?? maxWidth),
        "--quality",
        String(cfg.quality ?? jpegQuality),
        "--json",
      ];
      let completed;
      try {
        completed = await execFileFn(python, args, {
          encoding: "utf8",
          maxBuffer: 8 * 1024 * 1024,
          timeout: timeoutMs + 1500,
          signal,
          env: { ...env, G1_SENSE_FILTER: cfg.filter ?? env.G1_SENSE_FILTER ?? "" },
        });
      } catch (error) {
        const payload = lastJson(error?.stdout);
        throw new CameraError(payload?.error ?? `g1_u2_sense.py failed: ${error.message}`, error);
      }

      const payload = lastJson(completed.stdout);
      if (!payload || payload.ok !== true) {
        throw new CameraError(payload?.error ?? "g1_u2_sense.py returned invalid JSON");
      }
      if (typeof payload.image_b64 !== "string" || payload.image_b64.length === 0) {
        throw new CameraError("g1_u2_sense.py returned no JPEG image");
      }
      return normaliseFrame(payload);
    },
  };
}

// A 1x1 PNG, the same fixture the offline example uses: enough for a model prompt, nothing to look at.
const FIXTURE_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a90sAAAAASUVORK5CYII=";

/** `G1_FIXTURE_DEPTH="c,l,r[;c,l,r...]"` metres: one depth summary per frame, cycled. */
export function parseFixtureDepth(text) {
  return String(text ?? "").split(";").filter(part => part.trim()).map(part => {
    const values = part.split(",").map(item => Number(item.trim()));
    if (values.length !== 3 || values.some(value => !Number.isFinite(value))) throw new Error("G1_FIXTURE_DEPTH wants center,left,right in metres, frames separated by ;");
    const [center_m, left_m, right_m] = values;
    return { center_m, left_m, right_m, front_min_m: Math.min(center_m, left_m, right_m) };
  });
}

/**
 * In-process camera: serves the embedded PNG with a synthetic depth summary, cycling through
 * `frames`. A depth-only re-read returns the depth of the frame last served, as the replay worker
 * does, so the gate's live check sees what the model saw. Never touches a socket or a robot.
 */
export function createFixtureCamera({ frames = [{ center_m: 2.5, left_m: 2.5, right_m: 2.5 }], now = () => Date.now() } = {}) {
  if (!Array.isArray(frames) || frames.length === 0) throw new TypeError("fixture camera needs at least one frame");
  let index = 0;
  let last = frames[0];
  const summary = frame => ({
    center_m: frame.center_m, left_m: frame.left_m, right_m: frame.right_m,
    front_min_m: frame.front_min_m ?? Math.min(frame.center_m, frame.left_m, frame.right_m),
    center_valid_fraction: 1, front_valid_fraction: 1, scale_m: 0.001,
  });
  return {
    kind: "fixture",
    async capture(cfg = {}) {
      const capturedAt = now() / 1000;
      if (cfg.depthOnly === true) {
        return { imageB64: undefined, mimeType: "image/png", depth: summary(last), scene: null, capturedAt, width: 1, height: 1, jpegBytes: 0, timing: { socket_s: 0, encode_s: 0 } };
      }
      last = frames[index % frames.length];
      index += 1;
      const noDepth = cfg.includeDepth === false && cfg.needDepth === false;
      return { imageB64: FIXTURE_PNG, mimeType: "image/png", depth: noDepth ? null : summary(last), scene: null, capturedAt, width: 1, height: 1, jpegBytes: FIXTURE_PNG.length, timing: { socket_s: 0, encode_s: 0 } };
    },
  };
}
