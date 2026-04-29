# State: AI-Driven SEO Ranking Predictor

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-29)

**Core value:** Score maximally on the CIS 2450 rubric while satisfying NETS 1500's WWW + IR + Graph-Algorithms course-topic integration.
**Current focus:** Phase 1 — Data Layer (greenfield start)

## Active Milestone

**v1.0 Final Project Submission**
- NETS 1500 (85 pts) due **2026-04-29 23:59** — Phases 1-8
- CIS 2450 (143 pts) due **2026-04-30 23:59** — Phases 9-10

## Phase Status

| # | Phase | Status | Owner | Deadline Track |
|---|-------|--------|-------|---------------|
| 1 | Data Layer | ○ Pending | Rahil | NETS-tonight |
| 2 | Feature Engineering | ○ Pending | Rahil | NETS-tonight |
| 3 | Graph Layer | ○ Pending | Rahil | NETS-tonight |
| 4 | EDA + Charts | ○ Pending | Ayush | NETS-tonight |
| 5 | Modeling Sweep (8 models) | ○ Pending | Both | NETS-tonight |
| 6 | Recommendations + SHAP | ○ Pending | Ayush | NETS-tonight |
| 7 | Streamlit Dashboard | ○ Pending | Both | NETS-tonight |
| 8 | NETS 1500 Submission | ○ Pending | Both | **2026-04-29 23:59** |
| 9 | CIS 2450 Codebase Polish | ○ Pending | Both | CIS-tomorrow |
| 10 | Presentation + Final Submission | ○ Pending | Both | **2026-04-30 23:59** |

## Compliance Loop (CRITICAL)

After **every** phase, run:
```
mcp__project-compliance__full_compliance_check(project_summary=<current state>)
```

Save output to `.planning/compliance/phase-N-check.md`. Any P0 (MISSING / 🟡 partial in current phase scope) becomes a blocker added to current phase before advancing.

**Pre-submission final compliance check** is mandatory before both Gradescope submissions.

## Key Decisions Log

See `.planning/PROJECT.md` § Key Decisions. Update inline as decisions are made during execution.

**Pivot citation (must appear in README.md and data/README.md verbatim):** Ricky Gong's email of 2026-03-29 sanctions narrowed-domain pivot when full-scale collection is rate-limited. Direct quote in PROJECT.md § Context.

## Open Issues

- **CIS 2450 50K-row preference vs ~1500-page reality** — documented as known compliance risk; mitigated by (a) Ricky's email, (b) DATA-05 stretch to use (page, query) pairs ≈10K rows, (c) breadth/depth of analysis (graph + 8 models + dashboard polish).
- **Intermediate check-in attendance** — confirmed via Ricky's 2026-04-10 email (Google Meet link). Both members must verify they attended; loss = -5 pts per absent member.

## AI Usage Tracking (CIS 2450 §2.b — must be in deliverables)

Track per-phase what was AI-generated and what was hand-validated. Persist to `.planning/ai-usage-log.md`.

| Phase | AI-generated | Hand-validated | Validator |
|-------|-------------|----------------|-----------|
| 0 (planning) | PROJECT.md, REQUIREMENTS.md, ROADMAP.md scaffolding | Rubric items cross-checked against `full_compliance_check` output | Rahil |

This log feeds the readme.txt AI-usage section + dashboard About tab + presentation slide.

## Contribution Tracking (CIS 2450 hard requirement)

Both members must commit to git with their own attributed commits.

| Member | Email | GSD-tracked commits |
|--------|-------|---------------------|
| Rahil Patel | rahilp0920@gmail.com / rahilp07@seas.upenn.edu | 0 so far (current `git config user.email`) |
| Ayush Tripathi | tripath1@seas.upenn.edu | 0 so far — must `git config user.email tripath1@seas.upenn.edu` before committing his portions |

Atomic commits per task. Both members ship the Project Contribution Form at the end.

## Submission Checklist (final)

- [ ] NETS 1500 Gradescope: code zip + readme.txt + USER_MANUAL.md, both members added — by 2026-04-29 23:59
- [ ] CIS 2450 Gradescope: code zip + presentation/slides.pdf + presentation/recording.mp4, both members added — by 2026-04-30 23:59
- [ ] CIS 2450 Project Contribution Form (Google Form) — both members
- [ ] Final `full_compliance_check` returns no MISSING items
- [ ] Pre-submission smoke test: clean clone → `pip install -r requirements.txt` → `streamlit run src/dashboard/app.py` boots

## Next Step

Run `/gsd-plan-phase 1` to create the detailed plan for Phase 1 — Data Layer.

---
*State initialized: 2026-04-29*
