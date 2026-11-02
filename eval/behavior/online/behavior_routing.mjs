// Routing for the online agent's BEHAVIOR decisions (skill-level actions): when to deliberate and why.
// A history record is one executed skill: { action: { name, arguments }, info: { result, seconds, finish? } }.
// The reason sentence goes to the deliberate prompt as supervisor guidance.

// Skill results that mean the skill did not do what the model intended. Results are the runner's text: JSON for values
// (true / false / null / numbers / objects), "none" / "not found" from the perception tools, "ERROR ..." for exceptions,
// "REFUSED: ..." for a refused finish, "NO ACTION: ..." for a decision without a valid take_action.
const FAILED_VALUES = new Set(['false', 'none', 'not found']);

export function resultText(record) {
  const result = record?.info?.result;
  return typeof result === 'string' ? result : JSON.stringify(result ?? null);
}

export function refusedFinish(record) {
  return record?.action?.name === 'finish' && (record.info?.finish?.accepted === false || /^REFUSED/.test(resultText(record)));
}

export function failed(record) {
  const text = resultText(record).trim();
  return record?.action?.name === 'invalid_output' || refusedFinish(record) || /^(ERROR|NO ACTION)/.test(text)
    || FAILED_VALUES.has(text.toLowerCase());
}

// Arguments compared independently of key order.
function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
  return value;
}

export function sameAction(a, b) {
  return Boolean(a?.action && b?.action && a.action.name === b.action.name && a.action.name !== 'invalid_output'
    && JSON.stringify(canonical(a.action.arguments ?? {})) === JSON.stringify(canonical(b.action.arguments ?? {})));
}

function brief(record) {
  const text = resultText(record).replace(/\s+/g, ' ').trim();
  return `${record.action?.name}: ${text.length > 60 ? text.slice(0, 57) + '...' : text}`;
}

export function stallReason(context, patience = 3) {
  const history = context.history;
  const last = history.at(-1);
  if (!last) return null;
  const recent = history.slice(-patience);
  if (recent.length === patience && recent.every(failed)) {
    return `The last ${patience} actions did not do what you intended (${recent.map(brief).join('; ')}).`
      + ' Choose a different approach: another grasp candidate, the other arm, navigate closer, or re-detect the object.';
  }
  if (history.length >= 2 && sameAction(last, history.at(-2))) {
    return `Your last action repeated the previous one (${last.action.name} with the same arguments).`
      + ' Repeating a skill with unchanged arguments rarely changes the outcome; look at the current image and the results, decide what is still missing, and change something.';
  }
  if (refusedFinish(last)) {
    const arm = last.action.arguments?.arm;
    return `finish was refused: check_object_in_hand(${arm}) was False, so arm ${arm} does not hold the object and the episode continues.`
      + ' Check the image: re-grasp the object (sample_grasp_pose, then grasp_object with another candidate or the other arm) before calling finish again.';
  }
  return null;
}

export function shouldDeliberate(context, patience = 3) {
  return Boolean(stallReason(context, patience));
}
