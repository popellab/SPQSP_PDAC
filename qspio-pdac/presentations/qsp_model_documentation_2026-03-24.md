---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  :root {
    --color-bg: #fdfdfd;
    --color-fg: #1e1e2e;
    --color-accent: #2563eb;
    --color-muted: #64748b;
    --color-highlight: #dc2626;
    --color-surface: #f1f5f9;
  }
  section {
    background: var(--color-bg);
    color: var(--color-fg);
    font-family: 'Inter', 'Helvetica Neue', sans-serif;
    font-size: 28px;
    padding: 50px 70px;
    line-height: 1.5;
  }
  h1 {
    font-weight: 700;
    font-size: 44px;
    color: var(--color-fg);
    margin-bottom: 0.3em;
    letter-spacing: -0.02em;
  }
  h2 {
    font-weight: 600;
    font-size: 32px;
    color: var(--color-muted);
    margin-top: -0.2em;
  }
  h3 {
    font-weight: 600;
    font-size: 26px;
    color: var(--color-accent);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5em;
  }
  strong {
    font-weight: 600;
    color: var(--color-fg);
  }
  em {
    font-style: normal;
    color: var(--color-highlight);
    font-weight: 600;
  }
  ul, ol {
    padding-left: 1.2em;
  }
  li {
    margin-bottom: 0.4em;
  }
  li::marker {
    color: var(--color-accent);
  }
  blockquote {
    background: var(--color-surface);
    border-left: 4px solid var(--color-accent);
    border-radius: 0 8px 8px 0;
    padding: 0.8em 1.2em;
    margin: 1em 0;
    font-size: 26px;
    color: var(--color-muted);
  }
  blockquote strong {
    color: var(--color-fg);
  }
  table {
    font-size: 22px;
    border-collapse: collapse;
    width: 100%;
    margin: 0.5em 0;
  }
  th {
    background: var(--color-accent);
    color: white;
    font-weight: 600;
    padding: 10px 16px;
    text-align: left;
  }
  td {
    padding: 10px 16px;
    border-bottom: 1px solid #e2e8f0;
  }
  tr:nth-child(even) td {
    background: var(--color-surface);
  }
  code {
    background: var(--color-surface);
    padding: 0.15em 0.4em;
    border-radius: 4px;
    font-size: 0.9em;
  }
  pre {
    background: var(--color-surface);
    padding: 1em;
    border-radius: 8px;
    font-size: 20px;
  }
  .cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2em;
    align-items: start;
  }
  section.lead {
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
    padding: 60px 100px;
  }
  section.lead h1 {
    font-size: 52px;
    line-height: 1.2;
  }
  section.lead h2 {
    font-weight: 300;
    font-size: 28px;
    color: var(--color-muted);
    margin-top: 0.5em;
  }
  section.lead h3 {
    text-transform: none;
    font-size: 22px;
    color: var(--color-muted);
    font-weight: 400;
    letter-spacing: 0;
  }
  .dim {
    color: var(--color-muted);
  }
  .small {
    font-size: 20px;
    color: var(--color-muted);
  }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Documenting Model Development
## From version control to decision provenance

### QSP Meeting &nbsp;|&nbsp; March 24, 2026

---

# The Problem

Model development produces hundreds of changes over months — parameter values, structural additions and removals, design decisions informed by literature.

Without a record, reconstructing *why* the model looks the way it does requires archaeology: reading old code, searching emails, asking the person who may have graduated.

This matters at handoff, review, defense, and publication. The cost is deferred but real.

---

# A Progression, Not a Checklist

Different practices build on each other. Not everyone needs to be at the same level.

```
  Level 0    Code on local machine, no version control
     ↓
  Level 1    Git — code is tracked, changes are reversible
     ↓
  Level 2    Informative commits — reasoning captured in history
     ↓
  Level 3    Decision log — narrative record of modeling choices
     ↓
  Level 4    Parameter provenance — every value traceable to a source
     ↓
  Level 5    Model reference — living document of structure and caveats
```

Level 1 is the foundation. Everything else is optional but cumulative.

---

# Level 1: Why Git

Version control is the single highest-value practice for computational work.

- **Reversibility** — any change can be undone
- **History** — what the model looked like at any point in time
- **Collaboration** — multiple people can work on the same codebase
- **Backup** — code exists somewhere other than your laptop

If you are not using git, start here. Everything else in this talk assumes it.

---

# Level 1: Git as a Daily Habit

Git is most useful when committing is part of the daily workflow, not something done once a week or before a meeting.

A useful rule: **commit whenever something works**. Parameter changed and the simulation runs — commit. Module added and tests pass — commit. About to try something that may not work — commit first, so the previous state is recoverable.

```bash
git add parameters_PDAC.m fibroblast_module.m
git commit -m "Increased k_myCAF_death to 0.017 based on Kisseleva 2012"
git push
```

Frequent small commits produce a useful history. Infrequent large commits produce a history where every entry is "updated model."

---

# Level 1: Common Barriers

- "I'll break something" — version control makes changes reversible; this is the point
- "I don't know the commands" — `add`, `commit`, `push` covers 90% of daily use; alternatively, **GitHub Desktop** provides the same workflow through a graphical interface with no command line required
- "My code is messy" — tracked messy code is recoverable; untracked clean code is not
- "I'll learn it later" — the cost of not using version control compounds over time

---

# Level 2: Informative Commits

The difference between a useful and uninformative git history:

```
# Useless
updated model
fixed bug
changes

# Useful
Increased k_myCAF_death from 0.01 to 0.017 (Kisseleva 2012 Fig 3)
Removed PDGF species — undetectable by ELISA in WT PDAC (McCarty 2007)
Fixed Hill function: H_IL1_eff denominator was missing TGFb antagonism term
```

The second set requires 30 additional seconds per commit. Six months later, `git log --grep="PDGF"` recovers the full reasoning.

---

# Level 3: Decision Log

Some decisions span multiple commits or involve reasoning that doesn't fit a commit message — alternatives considered, dead ends, lessons learned.

A decision log is a single file (`DEVLOG.md` or `CHANGELOG.md`) where you append a few lines per work session.

```
## 2026-03-20 — Removed PDGF species from model

Wild-type PDAC cells produce undetectable PDGF-BB by conditioned
media ELISA (McCarty 2007). PDGF-B has a retention motif that
traps it in ECM. k_PDGF_sec was PRCC #1 — highest-sensitivity
parameter was uncalibratable.

Decision: remove species. Proliferation rates absorb basal PDGF.
```

---

# Decision Log: What Makes an Entry Useful

The diff shows *what* changed. The entry records *why*.

- The **reasoning** — why this change and not an alternative
- The **citation** — what literature informed the decision
- The **lesson** — anything that generalizes

```
k_psc_activation_myCAF showed a 15x discrepancy between targets.
The Bachem 1993 ODE had no Hill functions — k_act absorbed the
basal TGFb/PDGF Hill terms. After adding Hill gating to match the
QSP model's reaction structure, both targets agree on ~0.9/day.

Lesson: submodel forward models must replicate the QSP model's
parameterization, not simplify it.
```

This entry prevented repeating the same error two weeks later on a different parameter.

---

# Level 4: Parameter Provenance

Every non-default parameter value should be traceable to a source.

### Inline comment (minimal)

```matlab
p = addparameter(model, 'k_myCAF_death', 0.017, 'ValueUnits', '1/day');
% Kisseleva 2012 Fig 3: HSC fraction 14.3% -> 5.6% over 30 days post-CCl4
```

### Structured file

```yaml
k_myCAF_death:
  value: 0.017
  units: 1/day
  source: Kisseleva 2012, PNAS
  doi: 10.1073/pnas.1201840110
  evidence: "total HSC fraction drops from 14.3% to 5.6% over 30 days"
```

---

# Level 4: The "Estimated" Problem

Many QSP publications list parameter sources as "Estimated" in the supplement table.

This is not reproducible. There is no way for a reviewer to evaluate the estimate, for another group to reuse it, or for the original author to reconstruct the reasoning a year later.

Structured provenance replaces "Estimated" with a complete record: the source paper, the specific figure or table, the raw measurement, and how the parameter value was derived from it.

---

# Level 4: What Structured Provenance Enables

- **Calibration** — provenance files can be read programmatically into calibration and sensitivity analysis pipelines, linking each parameter directly to its constraining data
- **Reproducibility** — a reviewer or collaborator can independently verify every parameter value against its source
- **Publication** — the parameter table is a direct export rather than a manual reconstruction

---

# Level 5: Model Reference

A document describing the current state of the model — edited in place, not appended to.

```
The macrophage module modifies reactions from earlier modules:
- Appends *H_IL10 to Treg-mediated CD8 exhaustion reaction rate
- Creates IL-12, IL-10, IFNg species used by APC_module maturation

The APC module uses H_pore_APC defined in fibroblast_module. Since
APC_module is called first in assembly, this is a forward reference
resolved at simulation time.
```

Without this, adding or reordering modules silently breaks dependencies. Useful for model red teams — the reviewer can understand structure without a verbal walkthrough.

---

# What This Enables

From the last group meeting, the most requested formats were **model red teams**, **parameter/mechanism deep dives**, and **troubleshooting sessions**.

All three require that someone other than the author can understand the model quickly. Documentation at any level makes this easier:

| Level | What it enables |
|-------|----------------|
| Git history | Reviewer can see what changed and when |
| Informative commits | Reviewer can search for reasoning behind specific changes |
| Decision log | Reviewer can read the narrative arc of development |
| Parameter provenance | Reviewer can check sources without asking the author |
| Model reference | Reviewer can understand structure without running the code |

---

# Reducing Friction: Coding Agents

The main obstacle to documentation is that it competes with the actual work. LLM coding agents (Claude Code, Cursor, Copilot) can lower this cost substantially.

An agent that is already writing and modifying code can also:
- Generate commit messages from the diff, including reasoning it was part of
- Append decision log entries at the end of a work session
- Add parameter provenance comments inline as values are set
- Maintain a model reference document as modules change

The documentation becomes a byproduct of the development process rather than a separate task. The practices in this talk are all tool-agnostic — but an agent makes levels 2–5 nearly free.

---

# Discussion

- Where are you currently on the progression?
- What would help you move one level forward?

---

# Goal Setting

Each person: identify one concrete step to take in the next two weeks.

Examples:
- Put an existing project under version control
- Install GitHub Desktop and make a first commit
- Start writing one-sentence commit messages that include *why*
- Add a `DEVLOG.md` to a current project
- Add source comments to 10 parameters

Follow-up: **April 7** — brief check-in on progress and obstacles.
