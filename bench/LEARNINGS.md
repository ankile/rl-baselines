# Learnings / Guardrails

These notes encode project-specific expectations so future changes stay aligned with how this repo is actually used.

## Scope

- This is a private research harness for one active workflow.
- We optimize for fast iteration and reproducibility, not broad platform support.

## Defaults

- Prefer one clear path over many optional paths.
- Prefer explicit errors over implicit fallback behavior.
- Keep commands and configs concrete, not over-generalized.
- Assume `zsh` as the shell environment.

## Failure Philosophy

- If prerequisites are missing or state is inconsistent, fail loudly.
- Error messages should be direct and actionable.
- Do not hide setup/runtime failures behind warnings when runs would fail anyway.

## Practical Rule

When choosing between:
1. adding edge-case complexity for hypothetical users, or
2. keeping the path straightforward for current usage,

choose (2) unless there is a current, concrete need.
