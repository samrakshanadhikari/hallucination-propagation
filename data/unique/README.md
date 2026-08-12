# Unique fact datasets

Clean unique `(question, false_fact)` tables for the unified experiments.
These are **not** the old 1k-row scaffolds (which repeated the same facts with different metadata).

| File | Unique n | Notes |
|------|----------|--------|
| `geography_unique.csv` | 100+ | Strong-prior geography false facts |
| `math_unique.csv` | ~20 per category | numerical, attribution, wrong_statement, invented_theorem, fake_solved |
| `cutoff_unique.csv` | 91 | Post-cutoff fabricated AI/benchmark facts (all claims dated after 2025-08-31 for GPT-5.4-nano) |
| `all_unique_facts.csv` | combined | Convenience merge |

## Columns
- `question`, `false_fact`, `ground_truth` (always)
- `category` (math)
- `subcategory` (cutoff)
- `dataset` (`geography` / `math` / `cutoff`)

Existing unique facts from the original CSVs are included; new rows were added to hit agreed targets.
