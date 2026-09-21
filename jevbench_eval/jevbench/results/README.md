# Results

- `jevbench-v1-results.json` - the artifact. Aggregates only: no item text, no expected
  label, no per-item prediction. Every number carries the cohort and the denominator it
  was computed over, so a partial run can never be read as a full one.
- `charts/` - quality vs price, quality vs speed, and the calibration plot, drawn from
  that file.
- `../RESULTS.md` - the same numbers in prose, generated, not typed.

The artifact also carries two things that are deliberately kept *beside* v1 rather than
inside it: a `legacy_appendix` with older measurements of some of these models under a
different protocol, and a `repeatability` block from running one model over the whole
suite twice in one morning.
