# Security

Please report security issues privately through GitHub's "Report a vulnerability" (Security Advisories) on this
repository rather than in a public issue.

## Running REA safely

- **Generated code is not sandboxed.** Code-as-policy workers (`aloha-code`, the RoboTwin code policy) execute
  model-written Python in a separate process without the parent's credentials, but with the user's file-system and
  network access. Run them in a container or VM when the model or its inputs are untrusted.
- **The G1 service binds to `127.0.0.1` by default.** Set `G1_AGENT_TOKEN` before binding to another interface; every
  route except `GET /health` then requires `Authorization: Bearer <token>`.
- **Hardware starts in preview.** Robot adapters move hardware only with `execute: true` (and, for the G1 service,
  `G1_EXECUTE_ENABLED=1` plus an execute request). Keep an emergency stop within reach.
- **Credentials** come from the environment or a local `.env` (ignored by git); configuration files can reference them
  with `{ "env": "NAME" }`. Evaluation manifests drop `apiKey`, `password`, `authorization` and `accessToken` fields.
