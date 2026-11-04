# Final session and submission finalization

The project report is an English draft requiring group review. Do not label a live
session export complete: its final assistant response cannot appear until after it
has been sent. The current SESSION.md is a genuine provisional native export.

After this OpenCode session and both children are idle, from this repository run:

```bash
export UV_CACHE_DIR=.cache/uv
uv run python tools/export_sessions.py --sessions ses_e5d8a2028ffdpTf7nspYtJVp7I ses_e5d898135ffesfJ6YyK8ysLrvi ses_e5d893b81ffef5eNqrveRk54r2 --replace --final
```

The exporter requires idle sessions and coverage of session IDs referenced by the
parent, preserves native message/tool structure, and records message counts,
boundaries, model IDs and hashes of embedded redacted JSON. Inspect the export for
remaining secrets before committing. If it fails, retain the error and do not
invent missing transcript content. Then:

```bash
git add SESSION.md
git commit -m "report: finalize authentic parent and child session exports"
uv run python tools/clean_clone.py --seed 42
```

Only after this returns PASS and the group approves the report and submission scope
may the group create `git tag submission`. This repository covers Q2 only; coordinate
that tag with the shared project if other selected questions live elsewhere.
Earlier direct pytest temporary-directory compliance cannot be repaired retroactively;
see AUDIT.md. That limitation must remain disclosed at final acceptance.
