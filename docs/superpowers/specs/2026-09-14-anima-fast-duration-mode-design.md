# Anima Fast Training Duration Mode

## Scope

Fix issue #349 only for the Anima Fast engine. Do not change Kohya or other
training engines. Keep the existing Anima Fast `AdamW` schema and preset
defaults, and complete the same fallback at the backend adapter boundary.

## User Behavior

The Anima Fast form exposes a training duration mode with two choices:

- `Epoch`: selected by default and submits only `max_train_epochs`.
- `Steps`: submits only `max_train_steps`.

Import behavior follows the fields explicitly present in the imported TOML:

- Epoch only selects `Epoch`.
- Steps only selects `Steps`.
- Both fields without an explicit mode select `Epoch`, discard
  `max_train_steps`, and show a notice.
- A valid explicit `training_duration_mode` takes precedence for configs
  exported by the updated UI.
- Neither field keeps the default `Epoch` mode.

The TOML preview, copied/exported TOML, and `/api/run` payload must all contain
only the field selected by the active duration mode.

## Architecture

Add an Anima Fast-only UI field such as `training_duration_mode` to the schema.
Use schema field conditions so only the active duration input is displayed and
serialized. Keep the field itself UI-only so it is not forwarded upstream.

Centralize Anima Fast duration normalization in the frontend schema adapter.
The normalization accepts optional import metadata indicating whether
`max_train_epochs` and `max_train_steps` were explicitly present before schema
defaults were merged. Normal form edits use the selected mode directly.

The Anima Fast backend adapter is the final safety boundary:

- Epoch mode removes `max_train_steps`.
- Steps mode removes `max_train_epochs`.
- Legacy payloads without the UI mode infer Steps only when they contain
  `max_train_steps` and no `max_train_epochs`; otherwise they use Epoch.

This keeps old TOML files compatible while preventing ambiguous upstream
configuration.

## Error Handling

An imported legacy TOML containing both duration fields and no explicit mode is
accepted because older files may already contain both. The UI selects Epoch,
removes the steps value from the effective configuration, and displays an
informational notice. New-format configs with a valid explicit mode follow that
mode and do not display a contradictory notice.

Invalid or missing numeric values continue to use existing schema validation.
No new coercion rules are introduced.

## Tests

Frontend tests cover:

- Default Anima Fast model selects Epoch and serializes only epochs.
- Switching to Steps serializes only steps.
- Step-only import remains step-only after defaults are applied.
- Epoch-only import remains epoch-only.
- A two-field import selects Epoch and emits the conflict notice.
- Kohya serialization remains unchanged.

Backend tests cover:

- Explicit Epoch mode removes steps.
- Explicit Steps mode removes epochs.
- Legacy step-only payload remains step-only.
- Ambiguous legacy payload follows Epoch precedence.

Regression verification covers the real Anima Fast schema and all four Fast
presets continuing to default to `AdamW`.
