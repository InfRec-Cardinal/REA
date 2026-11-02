// Model-authored text that will be rendered into a later prompt: lessons, reflections, judgements,
// the deliberate engine's `next` line. All of it is data about the world, none of it may become an
// instruction to the next model. Two layers:
//   flattenLine      one line, printable, clipped — a newline or a control character could otherwise
//                    start a prompt line ("Advice: …") of its own
//   promptTextError  refuse text that reads as an instruction to the model rather than a lesson: a
//                    prompt label, "ignore the gate", "always output W", "you must …"
// The filters are deliberately narrow so the judges' own phrasing ("prefer Q", "turn Q before
// walking", "never pick W under 0.5 m", "you are 0.4 m from the wall") passes; a rejected text is
// dropped and counted, never rewritten into something that looks accepted. This is a blocklist of
// the label and directive forms a model plausibly emits, not a proof: a determined author can
// phrase an instruction it misses. What bounds the damage is elsewhere — a lesson can only ever bias
// which token is asked for, and the gate decides on live evidence regardless.

// C0/C1 controls, zero-width and bidi characters, BOM.
const CONTROL = /[\u0000-\u0008\u000b-\u001f\u007f-\u009f\u00ad\u200b-\u200f\u2028-\u202e\u2060-\u2064\ufeff]/gu;
const TOKEN = "[XWSADQEUPR]";
const COLON = "[:\uff1a]"; // ASCII or fullwidth

/**
 * Labels that head prompt lines. The ones that are also ordinary words a judge writes mid-sentence
 * ("centre depth: 0.42 m", "new plan: turn Q") only count at the start of the text, where a line
 * would begin; the ones that are never ordinary ("Advice:", "Token map:") count anywhere.
 */
const PROMPT_LABELS_ANYWHERE = new RegExp(`(^|[\\s(\\["'])(advice|experience|token map|live evidence|reviewer advice|recent steps|system|instruction|assistant|step \\d+ of \\d+)\\s*${COLON}`, "iu");
const PROMPT_LABELS_AT_START = new RegExp(`^(task|depth|plan|user|observation|observations|where|next token|answer|output|response|action)\\s*${COLON}`, "iu");

const DIRECTIVES = [
  // telling the model what to emit: an emission verb, or a label, followed by a token
  new RegExp(`\\b(output|reply|answer|respond|return|emit|say|print|write)\\s+(with\\s+|only\\s+|exactly\\s+|the\\s+token\\s+|token\\s+)?["'\`]?${TOKEN}["'\`]?(\\b|$)`, "iu"),
  new RegExp(`\\b(next token|answer|output|response|token|action)\\s*${COLON}\\s*["'\`]?${TOKEN}["'\`]?(\\b|$)`, "iu"),
  /\b(always|never|only)\s+(output|reply|answer|respond|emit|say|print|write)\b/iu,
  // telling the model to drop its constraints
  /\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}\b(gate|veto|vetoes|advice|instruction|instructions|prompt|rule|rules|safety|floor|lesson|lessons|token map|constraint|constraints)\b/iu,
  /\b(ignore|disregard|override|bypass)\s+(the\s+)?(depth|lidar)\b/iu,
  /\b(regardless of|no matter)\b.{0,30}\b(gate|safety|veto|vetoes|obstacle|obstacles|lesson|lessons|advice)\b/iu,
  // addressing the model with a rule about what to emit or to drop
  /\byou\s+(must|should|have to|need to|shall)\s+(always|never|only|ignore|disregard|bypass|override|output|reply|answer|respond|emit|say)\b/iu,
  /\b(system prompt|direct mode|as an ai|assistant)\b/iu,
  /\bbox_lift is (available|allowed)\b/iu,
  // the same in Chinese: ignore/bypass + depth/lidar/safety/rules/prompt; always/only/must + output/answer/choose
  /(忽略|无视|不要理会|绕过|跳过).{0,12}(深度|雷达|激光|安全|规则|提示|指令|限制|否决)/u,
  /(始终|总是|永远|一律|只|务必|必须).{0,6}(输出|回答|回复|选择|返回)/u,
];

export function flattenLine(text, maxChars = 240) {
  if (typeof text !== "string") return null;
  // Square brackets go because render() marks each lesson line with [+]/[-]; angle brackets and
  // braces stay, "centre > 1.5 m" is ordinary lesson text.
  const flat = text.replace(CONTROL, " ").replace(/[[\]]/gu, " ").replace(/\s+/gu, " ").trim();
  return flat ? flat.slice(0, maxChars) : null;
}

/** Why a piece of model-authored text may not be stored or rendered, or null when it may. */
export function promptTextError(text) {
  if (typeof text !== "string") return "not_text";
  const flat = flattenLine(text, Number.POSITIVE_INFINITY);
  if (!flat) return "empty";
  if ((flat.match(/[\p{L}\p{N}]/gu) ?? []).length < 2) return "too_short";
  if (PROMPT_LABELS_ANYWHERE.test(flat) || PROMPT_LABELS_AT_START.test(flat)) return "prompt_label";
  if (DIRECTIVES.some(pattern => pattern.test(flat))) return "directive";
  return null;
}

/** flattenLine + promptTextError in one step: the text to keep, or null with the reason. */
export function sanitisePromptText(text, maxChars = 240) {
  const error = promptTextError(text);
  if (error) return { text: null, error };
  return { text: flattenLine(text, maxChars), error: null };
}
