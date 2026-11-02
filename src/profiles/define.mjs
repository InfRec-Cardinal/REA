// A robot profile is the one place a robot's vocabulary lives: its tokens, what each token does,
// which tokens are motion, which undo each other, the arm interlock, the motion envelope and the
// sensing thresholds the gate applies. Every module that used to import G1 constants takes a
// profile instead and defaults to the G1 one, so a second robot is a second profile, not a fork.

export const PROFILE = Symbol.for("rea-agent.profile");

export class ProfileError extends TypeError {
  constructor(field, message) {
    super(`profile.${field}: ${message}`);
    this.name = "ProfileError";
    this.field = field;
  }
}

const VETO_DEPTH = new Set(["center", "left", "right"]);
const VETO_LIDAR = new Set(["center", "left", "right", "usable"]);
export const ROLE_NAMES = Object.freeze(["stop", "forward", "backward", "strafeLeft", "strafeRight", "turnLeft", "turnRight"]);

function deepFreeze(value) {
  if (value === null || typeof value !== "object" || Object.isFrozen(value)) return value;
  Object.freeze(value);
  for (const child of Object.values(value)) deepFreeze(child);
  return value;
}

const isToken = value => typeof value === "string" && value.length === 1 && value === value.toUpperCase();
const positive = value => Number.isFinite(Number(value)) && Number(value) > 0;

export function isProfile(value) {
  return Boolean(value) && typeof value === "object" && value[PROFILE] === true;
}

export function defineProfile(spec) {
  if (spec === null || typeof spec !== "object") throw new ProfileError("spec", "must be an object");
  if (isProfile(spec)) return spec;
  const {
    id, name = id, TOKENS, ACTION_TABLE, ACTIONS = [], roles = {}, motion = [], opposite = {}, arms = null,
    envelope = null, sensing = null, capabilities = {}, legacyDurationMaxMs = null, TOKEN_LEGEND = "",
  } = spec;
  if (typeof id !== "string" || !id) throw new ProfileError("id", "must be a non-empty string");
  if (!Array.isArray(TOKENS) || TOKENS.length === 0) throw new ProfileError("TOKENS", "must be a non-empty array");
  if (TOKENS.some(token => !isToken(token))) throw new ProfileError("TOKENS", "every token must be one upper-case character");
  if (new Set(TOKENS).size !== TOKENS.length) throw new ProfileError("TOKENS", "tokens must be unique");
  const tokenSet = new Set(TOKENS);
  const inTokens = (field, value) => {
    if (!tokenSet.has(value)) throw new ProfileError(field, `${JSON.stringify(value)} is not in TOKENS`);
  };
  if (ACTION_TABLE === null || typeof ACTION_TABLE !== "object") throw new ProfileError("ACTION_TABLE", "must be an object");
  for (const key of Object.keys(ACTION_TABLE)) inTokens(`ACTION_TABLE.${key}`, key);
  for (const token of TOKENS) {
    const record = ACTION_TABLE[token];
    if (!record || typeof record !== "object") throw new ProfileError(`ACTION_TABLE.${token}`, "is missing");
    if (record.token !== token) throw new ProfileError(`ACTION_TABLE.${token}.token`, `must equal ${token}`);
    if (typeof record.name !== "string" || !record.name) throw new ProfileError(`ACTION_TABLE.${token}.name`, "is required");
    if (typeof record.kind !== "string" || !record.kind) throw new ProfileError(`ACTION_TABLE.${token}.kind`, "is required");
    const veto = record.veto ?? { depth: null, lidar: null };
    if (veto.depth !== null && veto.depth !== undefined && !VETO_DEPTH.has(veto.depth)) throw new ProfileError(`ACTION_TABLE.${token}.veto.depth`, "must be center, left, right or null");
    if (veto.lidar !== null && veto.lidar !== undefined && !VETO_LIDAR.has(veto.lidar)) throw new ProfileError(`ACTION_TABLE.${token}.veto.lidar`, "must be center, left, right, usable or null");
    if (!sensing && (veto.depth != null || veto.lidar != null)) throw new ProfileError(`ACTION_TABLE.${token}.veto`, "must be null when the profile has no sensing");
    if (sensing && !sensing.depth && veto.depth != null) throw new ProfileError(`ACTION_TABLE.${token}.veto.depth`, "must be null when sensing.depth is null");
    if (sensing && !sensing.lidar && veto.lidar != null) throw new ProfileError(`ACTION_TABLE.${token}.veto.lidar`, "must be null when sensing.lidar is null");
  }
  if (!Array.isArray(ACTIONS)) throw new ProfileError("ACTIONS", "must be an array");
  if (roles === null || typeof roles !== "object") throw new ProfileError("roles", "must be an object");
  if (!tokenSet.has(roles.stop)) throw new ProfileError("roles.stop", "is required and must be in TOKENS");
  for (const [role, token] of Object.entries(roles)) {
    if (!ROLE_NAMES.includes(role)) throw new ProfileError(`roles.${role}`, `unknown role; use ${ROLE_NAMES.join(", ")}`);
    if (token != null) inTokens(`roles.${role}`, token);
  }
  if (!Array.isArray(motion)) throw new ProfileError("motion", "must be an array");
  for (const token of motion) inTokens("motion", token);
  if (motion.includes(roles.stop)) throw new ProfileError("motion", "must not include roles.stop");
  if (opposite === null || typeof opposite !== "object") throw new ProfileError("opposite", "must be an object");
  for (const [a, b] of Object.entries(opposite)) {
    inTokens("opposite", a);
    inTokens("opposite", b);
    if (opposite[b] !== a) throw new ProfileError("opposite", `${a} -> ${b} is not symmetric`);
  }
  if (arms !== null) {
    if (!Array.isArray(arms.holding)) throw new ProfileError("arms.holding", "must be an array");
    for (const token of arms.holding) inTokens("arms.holding", token);
    inTokens("arms.release", arms.release);
  }
  if (envelope !== null) {
    for (const key of ["minLinear", "minYaw"]) {
      if (!(Number.isFinite(Number(envelope[key])) && Number(envelope[key]) >= 0)) throw new ProfileError(`envelope.${key}`, "must be a non-negative number");
    }
    const range = envelope.pulseRangeS;
    if (!Array.isArray(range) || range.length !== 2 || !positive(range[0]) || !(range[1] >= range[0])) throw new ProfileError("envelope.pulseRangeS", "must be [min, max] seconds with 0 < min <= max");
  }
  if (sensing !== null) {
    if (sensing.depth) {
      for (const key of ["nearCenterM", "nearSideM", "floorCenterM", "floorSideM"]) {
        if (!positive(sensing.depth[key])) throw new ProfileError(`sensing.depth.${key}`, "must be a positive number");
      }
      if (sensing.depth.floorCenterM > sensing.depth.nearCenterM) throw new ProfileError("sensing.depth.floorCenterM", "must not exceed nearCenterM");
      if (sensing.depth.floorSideM > sensing.depth.nearSideM) throw new ProfileError("sensing.depth.floorSideM", "must not exceed nearSideM");
    }
    if (sensing.lidar) {
      if (typeof sensing.lidar.snapshot !== "string" || !sensing.lidar.snapshot) throw new ProfileError("sensing.lidar.snapshot", "is required");
      for (const key of ["maxAgeS", "nearCenterM", "nearSideM", "minPointsPerS"]) {
        if (!positive(sensing.lidar[key])) throw new ProfileError(`sensing.lidar.${key}`, "must be a positive number");
      }
      for (const key of ["center", "front", "left", "right"]) {
        const window = sensing.lidar[key];
        if (!Array.isArray(window) || window.length !== 2 || !(window[0] < window[1])) throw new ProfileError(`sensing.lidar.${key}`, "must be an ascending [from, to] azimuth window in degrees");
      }
    }
    const edges = sensing.buckets?.depth ?? [0.6, 1.5];
    if (!Array.isArray(edges) || edges.length !== 2 || !(edges[0] < edges[1])) throw new ProfileError("sensing.buckets.depth", "must be two ascending edges in metres");
  }
  if (legacyDurationMaxMs !== null && !(Number.isInteger(legacyDurationMaxMs) && legacyDurationMaxMs >= 0)) throw new ProfileError("legacyDurationMaxMs", "must be a non-negative integer or null");

  const legend = typeof TOKEN_LEGEND === "string"
    ? { full: TOKEN_LEGEND, short: TOKEN_LEGEND, min: TOKEN_LEGEND }
    : { full: TOKEN_LEGEND.full ?? "", short: TOKEN_LEGEND.short ?? TOKEN_LEGEND.full ?? "", min: TOKEN_LEGEND.min ?? TOKEN_LEGEND.short ?? "" };
  const actionSet = new Set(ACTIONS);
  const table = Object.fromEntries(TOKENS.map(token => [token, { ...ACTION_TABLE[token], veto: { depth: null, lidar: null, ...(ACTION_TABLE[token].veto ?? {}) } }]));
  const nameToToken = Object.fromEntries(TOKENS.filter(token => actionSet.has(table[token].name)).map(token => [table[token].name, token]));
  const stopRecord = table[roles.stop];

  function parseToken(value) {
    if (typeof value !== "string") return null;
    const token = value.trim().toUpperCase();
    return token.length === 1 && tokenSet.has(token) ? token : null;
  }
  function toToken(value) {
    if (value === null || value === undefined) return null;
    if (typeof value === "object") return toToken(value.token ?? value.action ?? null);
    const raw = String(value).trim().toUpperCase();
    if (raw.length === 1 && tokenSet.has(raw)) return raw;
    return nameToToken[raw] ?? null;
  }
  const actionForToken = value => {
    const token = parseToken(value);
    return token === null ? null : table[token];
  };
  const actionOrStop = value => actionForToken(value) ?? stopRecord;
  function isValidAction(value) {
    if (ACTIONS.length === 0 || legacyDurationMaxMs === null) return false;
    return value !== null && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 2
      && actionSet.has(value.action) && Number.isInteger(value.duration_ms) && value.duration_ms >= 0 && value.duration_ms <= legacyDurationMaxMs;
  }
  function legacyAction(token, durationMs = null) {
    const record = actionOrStop(token);
    const legacyName = ACTIONS.length === 0 || actionSet.has(record.name) ? record.name : stopRecord.name;
    const ms = Number.isInteger(durationMs) ? durationMs : record.defaultMs;
    return { action: legacyName, duration_ms: legacyDurationMaxMs === null ? ms : Math.max(0, Math.min(legacyDurationMaxMs, ms)) };
  }
  const motionFor = envelope === null ? null : function motionFor(token, durationMs = null, { maxDurationS = envelope.pulseRangeS[1] } = {}) {
    const record = actionOrStop(token);
    if (record.kind !== "velocity") return null;
    const requested = Number.isFinite(durationMs) ? durationMs / 1000 : record.durationS;
    const duration = Math.min(Math.max(requested, envelope.pulseRangeS[0]), maxDurationS);
    const scale = (value, floor) => (value === 0 ? 0 : Math.sign(value) * Math.max(Math.abs(value), floor));
    return {
      vx: scale(record.vx, envelope.minLinear),
      vy: scale(record.vy, envelope.minLinear),
      // a forward yaw trim is a correction, not a turn: it must not be inflated to the yaw floor
      yaw: record.yaw !== 0 && Math.abs(record.yaw) >= envelope.minYaw ? scale(record.yaw, envelope.minYaw) : record.yaw,
      duration: Number(duration.toFixed(3)),
    };
  };

  return deepFreeze({
    [PROFILE]: true,
    id,
    name,
    TOKENS: [...TOKENS],
    tokenSet,
    ACTION_TABLE: table,
    ACTIONS: [...ACTIONS],
    roles: { ...roles },
    motion: [...motion],
    opposite: { ...opposite },
    arms,
    envelope,
    sensing,
    capabilities,
    legacyDurationMaxMs,
    TOKEN_LEGEND,
    legend,
    HOLDING_TOKENS: arms ? [...arms.holding] : [],
    RELEASE_TOKEN: arms ? arms.release : null,
    BOX_LIFT: capabilities.boxLift ?? null,
    manipulation: capabilities.manipulation ?? null,
    MIN_LINEAR_SPEED: envelope?.minLinear ?? null,
    MIN_YAW_RATE: envelope?.minYaw ?? null,
    parseToken,
    toToken,
    actionForToken,
    actionOrStop,
    isValidAction,
    legacyAction,
    motionFor,
  });
}
