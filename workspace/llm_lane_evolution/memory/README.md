# LLM Lane Common Memory

This directory is the compact shared memory for current and future BonsaiEvolve agents.

## Design

`reports.csv` is the single append-only index. One row represents one completed agent report/candidate attempt. The row contains only high-value metadata, a compact result summary, the main reusable lesson, and immutable pointers to the detailed committed evidence.

Detailed reports are not copied here. A pointer such as:

```text
git:58e8dbf6188bf89d2b24aaa922498764fb5b815b:workspace/llm_lane_evolution/r1/a1/proposal.md
```

means:

```bash
git show 58e8dbf6188bf89d2b24aaa922498764fb5b815b:workspace/llm_lane_evolution/r1/a1/proposal.md
```

This keeps the common memory small even when the project accumulates hundreds of thousands of reports.

## Agent workflow

Before proposing a new mechanism, query the memory for the current lane and related mechanism keywords. Read full historical reports only when a compact row is directly relevant.

Examples:

```bash
.venv/bin/python tools/llm_lane_memory.py tail --limit 20
.venv/bin/python tools/llm_lane_memory.py search --lane market_realization_mechanics --limit 30
.venv/bin/python tools/llm_lane_memory.py search --text "market saturation" --limit 20
.venv/bin/python tools/llm_lane_memory.py search --build-improvement yes --limit 30
.venv/bin/python tools/llm_lane_memory.py show r1-a3-ba253b5c
.venv/bin/python tools/llm_lane_memory.py validate
```

The query tool streams the CSV instead of loading it wholesale. `tail` keeps only the requested number of rows in memory. Search is O(number of rows) time and O(result limit) memory, which is sufficient for hundreds of thousands of compact records without introducing a database.

## Append policy

The memory is maintained after each completed agent report. Keep rows compact and factual. Do not paste raw traces or long reports into CSV fields.

Required discipline:

- `report_id` is unique and stable.
- `report_commit` is the immutable commit containing the detailed report/evidence.
- `compact_summary` records the decisive result, not the whole narrative.
- `key_lesson` records what future agents should do or avoid.
- `proposal_ref`, `summary_ref`, and `metrics_ref` use `git:<commit>:<repo-path>` when those files live on an experimental branch rather than `main`.
- `tags` is a semicolon-separated keyword set for cheap retrieval.
- Do not rewrite old rows merely because later interpretation changes. Add a new report row for a new experiment.

This memory is an index, not an evaluation oracle. Agents must still inspect actual source/evidence and exact simulator results before reusing a mechanism.
