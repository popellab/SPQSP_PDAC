# Model debugging wisdom

Lessons learned from iterative debugging sessions. Each section captures a
trap we've actually fallen into and the rule that prevents the next session
from repeating it. Companion to `iterative_model_debugging.md`, which covers
the *mechanics* (worktree, build, trace, regen, SBI) — this doc covers the
*reasoning* you bring to those mechanics.

When you find yourself stuck or your fix isn't working, scan this list before
proposing the next change.

---

## 1. Know which "prior" you're actually changing

There are at least four separate places a parameter's distribution is set,
and they shadow each other in non-obvious ways. Before editing anything, find
out which one is binding for the parameter you're chasing.

**The hierarchy** (lowest precedence first):

1. **`parameters/parameters_PDAC.m` — `params.<name>.Value`**: the deterministic
   default used to write `resources/cpp/param_all.xml`. This is the value the
   `qsp_sim` binary uses for hand-edited XML traces; it is *not* what SBI
   samples around.
2. **`parameters/pdac_priors.csv` — `dist_param1`, `dist_param2`**: the
   lognormal `(μ, σ)` that SBI samples from for params with **no submodel
   target**. This is the right knob for params that are CSV-only.
3. **`notes/calibration/submodel_priors.yaml` — `marginal.{mu,sigma,...}`**:
   the inferred posterior over a parameter from its submodel target's data.
   For any parameter with an entry here, the SBI runner uses **this**
   distribution and ignores the CSV row entirely. Verify with
   `grep -A 6 "name: <param>$" notes/calibration/submodel_priors.yaml`.
4. **`calibration_targets/submodel_targets/*.yaml` — the data**: literature
   measurements that, after MCMC inference, produce the entry in (3).
   Editing these YAMLs to "widen σ" is *prior tuning by another name* and
   not allowed (see `feedback_submodel_targets_are_data.md`).

**The check that catches CSV-shadowing silently:** if you edit a CSV σ for a
param with a submodel entry and re-run SBI, `figures/.../artifacts/theta_train.pt`
will be **bit-identical** to the pre-edit run. The pool's `config_hash`
changes (CSV bytes are hashed), so the cache key looks fresh — but the
samples are the same. md5 the theta tensors before believing the σ change
took effect.

> Memory: `feedback_csv_sigma_shadowed_by_submodel.md`,
> `feedback_submodel_targets_are_data.md`

---

## 2. Trace baseline ≠ SBI sampling baseline

`resources/cpp/param_all.xml` is generated from CSV `params.<name>.Value`
defaults. When you run `qsp_sim` with a hand-edited XML, every parameter
*you don't change* sits at its CSV default. SBI does **not** sample around
those defaults — for any parameter with a submodel entry, it samples around
the submodel posterior median, which can be shifted off the CSV default.

If your trace is meant to predict what SBI will explore, **overlay submodel
medians onto `param_all.xml` first**. The runbook §3 has the snippet.

> A trace that shows "k_X = 2.0 reaches the target" from the CSV-default
> baseline does *not* imply SBI can reach the target. It only proves that
> the model's deterministic forward map can produce that value somewhere in
> parameter space. It says nothing about reachability in the actual SBI prior.

---

## 3. The qsp_sim binary can silently lie

After a `git pull`, after a merge, after switching branches that touched
`cpp/qsp/ode/` or `parameters_PDAC.m` — **rebuild the binary** before
trusting traces. A stale binary built against old parameter names will
silently zero out species when fed XML with the new names (no error, just
junk data: V_T.CD8 = 1e-34 or similar).

The diagnostic: if a baseline trace gives values that disagree with a
recent prior PC by 10⁻³⁰ or so, suspect a stale binary first. Rebuild via
`cmake --build cpp/sim/build --target qsp_sim` and re-run.

The setup script seeds the worktree's `_deps/` to skip the SUNDIALS clone,
but doesn't actually build the binary — that's a separate step.

> See `feedback_no_redundant_refresh_cpp.md` for when *not* to refresh.

---

## 4. Single-knob deterministic traces can mislead about joint SBI behavior

A deterministic trace that shows "param_X × 10 → density target" is a
*reachability* result, not an *SBI behavior* prediction. SBI samples
**jointly** from a many-dimensional prior. The marginal probability of any
one knob being at +1.5σ is fine; the joint probability of three knobs being
simultaneously at +1.5σ in the right direction is small.

Even if a trace shows the model can reach the target, the prior PC may not
put much mass there because the favorable corner requires multiple
correlated extreme draws. This is exactly why we keep finding "the model
can reach the target" in deterministic traces but SBI keeps missing.

**Mitigation:** when a trace says "param_X reaches target," also ask:
- What's the +1σ deviation needed?
- How many other params need to co-move?
- Is there a way for the prior to *jointly* sample the favorable corner?

If the joint probability of the favorable corner is small, you need either
a structural change (so the corner is reachable from a single-knob lift) or
a calibration target re-examination (so the goalpost is closer to the prior
center).

---

## 5. Tightly coupled targets create degenerate attractors

Multiple targets on **fractions** (treg_fraction, mycaf_fraction,
cd8_exhausted_fraction) with sharp CIs can pull the posterior into a
degenerate corner where all densities collapse to ~zero and the fractions
are trivially satisfied at 0/0 ≈ whatever-you-want.

**Symptom:** prior PC looks fine, but posterior PC has elasticities >100,
some observables collapse to 1e-29, and the conflict pairs all anchor on
the same fraction target.

**Diagnosis:** look at `predictive_summary.csv` — if `post_med_raw` for
density observables has shrunk by 10⁹ from prior_med while fraction
observables have moved *toward* their targets, you're in the trap.

**Mitigation:** don't add more fraction targets without thinking about
whether they collapse to a degenerate corner. When they exist, either
(a) widen their CI95 enough that the degenerate corner isn't *exactly*
satisfied, or (b) add density anchors that exclude the corner, or (c) raise
cohort-RE τ so the posterior tolerates fraction-target misses.

---

## 6. Positive feedback loops don't ignite at baseline

Several mechanisms in the model (chemokine recruitment, antigen-driven
proliferation, IFNg-induced gene programs) are positive-feedback loops
gated by upstream activity. At a cold-start immune-excluded baseline:

- PD-1/PDL-1 is fully saturated (`H_PD1_C1 ≈ 0.997`) → CD8 killing ≈ 0
- → IFNg production ≈ 0 (`V_T.IFNg ≈ 10 fM`, vs Hill EC50 ~10 pM)
- → CXCL9 induction ≈ 0 (Hill at 0.1%)
- → CXCR3 chemotaxis ≈ 0
- → CD8 recruitment stays at the constitutive `q_CD8_T_in` rate

Activating the mechanism (e.g., enabling `chemotaxis_module`) doesn't help
the baseline because the loop is dormant. The mechanism is a *secondary
amplifier* for already-inflamed tumors (post-checkpoint-blockade, GVAX), not
a primary igniter from cold.

**Implication for debugging:** a structural addition that makes biological
sense for treatment scenarios may have zero impact on baseline observables.
Trace the upstream Hill values first — if they're at <1% engagement, the
new mechanism is dormant and won't help the target you're chasing.

---

## 7. Trace the actual species state before proposing a fix

Before suggesting "widen σ on X" or "add mechanism Y," dump the trace state
at submodel medians and look at:

- All Hill values (`H_PD1_C1`, `H_TGFb_Teff`, `H_pore_T`, `H_stiff_Texh`,
  `H_CXCL12_Texcl`, `H_P1`, ...) — are any saturated (→1) or zeroed (→0)?
- Key cytokines (`V_T.IFNg`, `V_T.TGFb`, `V_T.CXCL12`) — are they at the
  EC50s of their downstream Hills?
- Compartment populations (`V_T.CD8`, `V_C.CD8`, `V_LN.mcDC1`,
  `V_LN.nCD8`) — where in the supply chain is the bottleneck?
- ECM state (`phi_collagen`, `d_pore`, `stiffness`) — is desmoplasia
  saturating the pore/stiffness Hills?

A saturated upstream gate is the most common "structural problem" — adding
mechanism downstream of a saturated gate doesn't unstick anything.

> Memory: `feedback_trace_before_proposing_fix.md`

---

## 8. Multi-source the calibration target before tuning the model

A single-source calibration target may be a literature outlier. Before
proposing structural changes to bring the model up to a target, check
whether other studies report values closer to what the model produces.

**Specific case (cd8_density_baseline):** Jansen 2021 (TMA, n=444) reports
median 138 cells/mm². Sereti 2022 HR-intact (n=166) reports mean 40.5
± SD 50.9, and Wang 2015 (n=92) reports ~67 (HPF-converted). The pooled
multi-source central tendency is ~70-90, much closer to what the model
reaches at +2σ-on-all-CD8-knobs (112).

The calibration target is a goalpost, and goalposts can be wrong. Re-source
before re-engineering.

---

## 9. Submodel targets are data, not knobs

When SBI can't reach a target for a parameter that's submodel-anchored,
the legit moves do **not** include "widen the σ on the submodel target." The
σ in `submodel_priors.yaml` is the NPE-derived posterior over the data; the
data has its own uncertainty model and can't just be loosened to give SBI
flexibility.

What you *can* do:

1. **Question the data.** Is the experimental context appropriate for the
   in vivo PDAC scenario the realized observable measures? If not, the
   submodel target may need replacing or augmenting — that's an extraction
   task.
2. **Multi-source the realized calibration target** (the observable, not
   the submodel target).
3. **Edit `pdac_priors.csv` σ for params with NO submodel entry.**
4. **Add structural mechanism** if the issue is a missing pathway.
5. **Accept the gap** and let cohort-RE τ absorb it.

> Memory: `feedback_submodel_targets_are_data.md`,
> `feedback_no_prior_tuning.md`

---

## 10. Don't fabricate citations on parameter values

When defining new parameters or commenting on existing ones, only cite
papers you've verified contain the specific value or range you're
attributing. If you used a value because it "felt reasonable" or because a
similar parameter in the model uses a similar number, mark it `ESTIMATED`
in the Notes and `derivedFrom = {'estimated'}`. A plausible-looking PMID
attached to an unverified value is worse than no citation at all — it
launders an estimate as a measurement.

> Memory: `feedback_no_fabricated_citations.md`

---

## 11. Reachability ≠ prior PC coverage

A target can be "reachable" (some draw in the prior produces it) without
being well-covered in the prior PC marginals. Prior PC misses are not
necessarily structural failures — if the obs is reachable in the tail, the
posterior can still pull it.

The structural test is: is the obs reachable under *any* combination of
prior draws? If not, no amount of NPE training will close the gap.

> Memory: `feedback_reachability_not_prior_coverage.md`

---

## 12. When writing a calibration target with `distribution_code`, run it before populating `empirical_data`

The `empirical_data.median` and `empirical_data.ci95` fields must match what
`distribution_code` actually produces from the listed `inputs`. Hand-writing
those fields by eye produces silently-wrong targets — the schema validator
won't catch a 30% error in median because both numbers are well-formed.

Always compute the output of your `distribution_code` (paste the snippet
into a python REPL with the same RNG seed and inputs) and paste those exact
empirical values into the YAML. The numbers should match to 3-4 sig figs.

This is especially important when the distribution involves lognormal
moment-matching from (mean, SD) — the resulting median is `exp(mu_log) =
mean^2 / sqrt(mean^2 + SD^2)`, not the mean. Eye-estimated medians for
heavy-tailed distributions are usually wrong by 1.5-2×.

---

## 13. The audit script is for humans; agents use regen_submodel_priors

`scripts/parameter_audit.py` is a user-facing review tool that produces a
Markdown report and PPC tables. Agents working in the iterative debug loop
should use `scripts/regen_submodel_priors.py [--invalidate PARAM ...]`
instead — same component-inference machinery, no PPC, no Markdown, faster.

> Memory: `feedback_parameter_audit_for_inference.md`

---

## Recommended debug ordering

When SBI flags a target as misspecified or unreachable:

1. **Identify whether the constraining parameter has a submodel entry.**
   Determines which prior you can actually move.
2. **Trace the model state at submodel medians** (with the SBI-baseline
   overlay snippet from runbook §3). Dump all Hill values and key species.
3. **Identify the upstream gate.** Saturated PD-1, dormant chemokine loop,
   choked priming pipeline?
4. **Sweep the candidate knob deterministically.** Confirm the model can
   reach the target somewhere.
5. **Check joint reachability.** Even if single-knob works at +1σ, can
   the prior actually sample that corner with reasonable probability?
   (The prior PC quantiles tell you.)
6. **Check whether the target itself is robust.** Multi-source the
   literature; the goalpost may be wrong.
7. **Decide where to invest:**
   - CSV σ widen → cheap, only works for unanchored params.
   - Multi-source target → cheap, shifts goalpost to consensus.
   - Structural addition → expensive, may not help baseline if upstream
     gates are saturated.
   - Cohort-RE τ → cheap, hides the gap rather than fixing it.

Don't stop at step 4. Most failed fixes today were "trace shows it's
reachable" + "let's edit the prior" without confirming the fix can
actually flow through joint SBI sampling.
