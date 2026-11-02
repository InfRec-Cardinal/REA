export class ActionError extends Error {
  constructor(message) {
    super(message);
    this.name = "ActionError";
    this.code = "invalid_action";
  }
}

function validateValue(value, schema, path) {
  if (schema.enum && !schema.enum.some(candidate => JSON.stringify(candidate) === JSON.stringify(value))) {
    throw new ActionError(`${path} must be one of ${JSON.stringify(schema.enum)}`);
  }
  switch (schema.type) {
    case "object": {
      if (value === null || typeof value !== "object" || Array.isArray(value)) throw new ActionError(`${path} must be an object`);
      for (const required of schema.required ?? []) {
        if (!Object.hasOwn(value, required)) throw new ActionError(`${path}.${required} is required`);
      }
      for (const [name, argument] of Object.entries(value)) {
        const property = schema.properties?.[name];
        if (property) validateValue(argument, property, `${path}.${name}`);
        else if (schema.additionalProperties === false) throw new ActionError(`${path}.${name} is not supported`);
      }
      break;
    }
    case "array":
      if (!Array.isArray(value)) throw new ActionError(`${path} must be an array`);
      if (schema.minItems !== undefined && value.length < schema.minItems) throw new ActionError(`${path} needs at least ${schema.minItems} items`);
      if (schema.maxItems !== undefined && value.length > schema.maxItems) throw new ActionError(`${path} allows at most ${schema.maxItems} items`);
      value.forEach((item, index) => {
        const itemSchema = schema.prefixItems?.[index] ?? schema.items;
        if (itemSchema === false) throw new ActionError(`${path}[${index}] is not supported`);
        if (itemSchema && itemSchema !== true) validateValue(item, itemSchema, `${path}[${index}]`);
      });
      break;
    case "integer":
    case "number":
      if (!Number.isFinite(value) || (schema.type === "integer" && !Number.isInteger(value))) throw new ActionError(`${path} must be a ${schema.type}`);
      if (schema.minimum !== undefined && value < schema.minimum) throw new ActionError(`${path} must be >= ${schema.minimum}`);
      if (schema.maximum !== undefined && value > schema.maximum) throw new ActionError(`${path} must be <= ${schema.maximum}`);
      if (schema.exclusiveMinimum !== undefined && value <= schema.exclusiveMinimum) throw new ActionError(`${path} must be > ${schema.exclusiveMinimum}`);
      if (schema.exclusiveMaximum !== undefined && value >= schema.exclusiveMaximum) throw new ActionError(`${path} must be < ${schema.exclusiveMaximum}`);
      break;
    case "string":
      if (typeof value !== "string") throw new ActionError(`${path} must be a string`);
      break;
    case "boolean":
      if (typeof value !== "boolean") throw new ActionError(`${path} must be a boolean`);
      break;
    case undefined:
      break;
    default:
      throw new ActionError(`unsupported action schema type: ${schema.type}`);
  }
}

export function validateAction(action, actions) {
  const specification = actions.find(candidate => candidate.name === action?.name);
  if (!specification) throw new ActionError(`action ${JSON.stringify(action?.name)} is not available in this observation`);
  const normalized = { name: action.name, arguments: action.arguments ?? {} };
  validateValue(normalized.arguments, specification.parameters ?? { type: "object", properties: {}, additionalProperties: false }, "arguments");
  return normalized;
}
