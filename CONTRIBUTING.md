# Contributing

VisualTaste is a personal research project, but issues and pull requests are
welcome: bug reports, clearer docs, new rungs, new simulator scenarios.

## Setting up

Follow the README's [Quick start](README.md#quick-start). You do not need a GPU
or the full collection to work on the code: every test runs on a CPU with
synthetic fixtures, and CI runs them on Ubuntu with CPU-only PyTorch.

## Before you open a pull request

```bash
just check          # or ./scripts/check.sh  ·  Windows: .\scripts\check.ps1
```

That runs everything CI runs: `ruff check`, `ruff format --check`, pytest,
the TypeScript type check, Vitest, the production build and Playwright.

## Ground rules

- **Never change a cached result silently.** A change to existing scoring,
  ranking or simulator code must leave every cached study bit-identical, or
  say clearly in the pull request which results it changes and re-run them.
  New behaviour goes behind a new `RungSpec` option (`eval/study.py`), which
  gets its own cache key.
- **Results are appended, not edited.** `docs/EVALUATION.md` records each study
  with its command and date. Re-running a study appends a new section.
- **Record decisions.** A non-obvious design choice gets a numbered entry in
  `docs/DECISIONS.md`: what was chosen, what else was considered, and why.
- **Nothing leaves the machine.** The app binds to `127.0.0.1` and makes no
  external requests at runtime. Do not add analytics, telemetry or CDN assets.
- **Every image is credited.** Do not add a source whose licence the licence
  gate cannot record (`data/sources/`).
- **Tests come with the change.** Maths gets a hand-worked example; a bug fix
  gets the test that would have caught it.

## Adding a recommender rung

1. Subclass `Recommender` (`rec/base.py`): `_fit`, `score`, and optionally
   `rerank` and `explain`.
2. Register it in `api/deps.load_rung` and give it a `RungSpec` name.
3. Add a study preset in `jobs/evaluate.py` comparing it with the rung below,
   with paired seeds, and record the result in `docs/EVALUATION.md`.

## Reporting a bug

Use the bug report template. Include the command, the full error, your OS,
and whether CUDA is available (`python -c "import torch; print(torch.cuda.is_available())"`).
