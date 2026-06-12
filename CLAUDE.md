# scgrad — repo conventions

## Authorship

This repository ships under the maintainer's identity (g-lanza). No AI
attribution anywhere: no "Co-Authored-By" trailers, no "Generated with"
footers, no AI mentions in commit messages, docs, or metadata. A commit-msg
hook in `.githooks/` strips such trailers; `core.hooksPath` points at it.

## Commit messages

Conventional-commit style: `feat:`, `fix:`, `test:`, `docs:`, `chore:`,
`refactor:`. Imperative mood. Plain. Describe the change, not the author.
No emoji.

## Code rules

- No stubs, no TODOs, no `NotImplementedError`, no empty `pass` bodies.
  Every public function runs end-to-end.
- No emoji, marketing language, or decorative comments. Docstrings say what
  a thing does and note its gotchas (scale factors, correlation ids).
- Every `torch.autograd.Function` passes `torch.autograd.gradcheck` in
  float64 at atol 1e-4 before it counts as done.
- The dual-path test (`tests/test_dual_path.py`) is the conscience of the
  repo: the differentiable path and the bit-accurate path must converge as
  bitstream length grows. Never loosen its tolerance to get green.
- Bipolar multiply is XNOR (`v = v_a * v_b` in value space). MUX addition is
  a scaled average, not a sum; the scale factor is tracked on `SCNumber`.
- `ruff check`, `ruff format --check`, and `mypy src` (strict) must be clean
  before any commit.

## Layout

src layout (`src/scgrad/`), tests import the installed package. Use `uv` for
everything: `uv sync`, `uv run pytest`, `uv build`.
