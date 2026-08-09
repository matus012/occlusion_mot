# blockers — 100_occlusion_mot

> `refactored_method.md` §4: **there is no `AskUserQuestion`.** A blocker is a row here, and then
> **work continues on everything not blocked.** The loop never stops to ask.
>
> A row is only for the five things that actually require the operator — force-push, deleting
> repos/branches, making a repo public, spending money, wiping an environment — or for a genuine
> external dependency (dataset access, hardware, a threshold change). Ambiguity is **not** a
> blocker: implement the reading you can defend and state the assumption.

Every row carries a **recommendation**. A row that presents options without one is asking a
question with extra steps.

Created 2026-08-09 by the infra audit: this project's three agents were routed to
`reports/blockers.md` by `ws/CLAUDE.md` and by their own definitions, and the file did not exist.

| date | task | what is blocked | what was tried | what is needed | recommendation | state |
|---|---|---|---|---|---|---|
| — | — | *(none open)* | — | — | — | — |

## Resolved

Keep them. A resolved row is the record of who decided what.
