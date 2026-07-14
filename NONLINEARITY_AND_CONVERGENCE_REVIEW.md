# Review of system nonlinearity and nonlinear convergence

## Scope and main conclusion

This review covers `npsat_flow_assemble.impl.h`, the nonlinear loop in `npsat_v2.cpp`, head recovery and convergence in `npsat_flow_post.impl.h`, and head-dependent well conductance in `mnwells.h`. No source code was changed.

The inner linear solve succeeds on every reported nonlinear iteration (321--447 CG iterations). The failure is in the outer frozen-coefficient fixed-point iteration. The run initially contracts rapidly, then reaches an oscillatory floor of roughly 2--3 m, far above the 0.0172 m threshold. This is characteristic of a local coefficient or active-set oscillation, rather than failure of the linear solver.

The four important nonlinear mechanisms are:

1. relative conductivity `r(h)` changes the flux matrix;
2. effective storage `S_eff(h)` changes both the condensed operator and right-hand side;
3. recharge routing selects a head-dependent receiver cell;
4. well-link conductance changes with wetted screen length.

## 1. Where nonlinearity enters the assembled equations

For a fixed head guess `h*`, assembly solves a linear hybrid mixed system:

```text
[ A(h*)      B^T       C^T ] [ Q      ]   [ 0 ]
[ B        M(h*)/dt      0 ] [ H      ] = [ F ]
[ C           0          0 ] [ Lambda ]   [ g ]
```

After local condensation, in abbreviated form,

```text
Kc(h*)    = M(h*) - dt B A(h*)^-1 B^T
E(h*)     = B A(h*)^-1 C^T
S_hat(h*) = C A(h*)^-1 C^T - dt E^T Kc^-1 E + GHB
V(h*)     = M(h*) H_old + dt F
b_hat(h*) = -g - E^T Kc^-1 V.
```

Well terms further modify `S_hat`, trace/well couplings, the well block, and both right-hand sides. Solving this frozen problem defines a raw candidate `h_new = G(h*)`. Ordinary damped Picard accepts

```text
h_accepted = (1-omega) h* + omega G(h*).
```

The present convergence test is applied to the **raw fixed-point residual** `G(h*)-h*`, before damping or Anderson. This is valid, but logs should distinguish it from the accepted step.

### 1.1 Relative conductivity

For cell thickness `b=z_top-z_bot`, the code uses

```text
theta_raw = (h-z_bot)/b

dry cell:                 r = r_min
partially saturated cell: r = max(r_min, theta_raw)
saturated interior cell:  r = 1
top-layer cell:           r = max(r_min, theta_raw), possibly greater than 1.
```

Assembly applies `K_eff=rK`, so `A(h)` contains `(1/r)K^-1`. The dependence therefore propagates through `A^-1` into every condensed matrix, right-hand side, recovered head, and flux. The piecewise rule has kinks at the dry/partial and partial/saturated transitions. These kinks can destroy Picard contractivity when a head crosses a cell face.

The default `r_min=0.1` is numerically stabilizing but physically large: a dry cell retains 10% conductivity. Reducing it blindly is not advised because this also worsens conditioning. It should be tested only after the oscillating cells are identified.

For top-layer cells, allowing `r>1` means transmissivity exceeds `K` times the geometric cell thickness. This modeling choice should be verified explicitly.

### 1.2 Effective storage

The code computes

```text
theta    = clamp((h-z_bot)/b, 0, 1)
S_conf   = Ss theta
S_yield  = Sy/b
w(h)     = sigma((h-z_bot)/eps) sigma((z_top-h)/eps)
S_eff(h) = S_conf + w(h) S_yield.
```

This is smoother than an on/off specific-yield term, but can still be stiff. The default `eps=0.05 m` is an absolute thickness. In cells several metres thick, the large `Sy/b` contribution transfers over a narrow head interval. When the water table crosses a layer face, the specific-yield window can transfer between adjacent cells and create a two-cell oscillation.

Storage affects both `Kc` and `V=M H_old+dt F`; it is not only a right-hand-side coefficient. A formulation issue also deserves verification against the theoretical derivation: with head-dependent capacity, a conservative transient equation is normally based on an increment of stored water. Using `M(h_guess) H_old` may not represent that increment. A one-column water-balance benchmark is needed before drawing a final conclusion.

### 1.3 Recharge receiver selection

Recharge is routed downward until a cell satisfies

```text
psi = h-z_bot > max(eps, selected_saturated_fraction*b)
r > RechargeMinRelativeK.
```

Different wetting and drying fractions provide hysteresis, but the chosen receiver remains a discrete active set. Moving it one layer changes the face where recharge enters and therefore the spatial right-hand side.

The reported recharge total changes by only about 46 flow units between iterations, but this does not exclude strong local rerouting: several columns can switch receivers while their global totals nearly cancel. Changed receiver IDs and rerouted area are more informative than the aggregate flux.

For a receiver, the code calculates `effective_z_top=max(h,z_top-0.1b)` and calls the coefficient routine again. However, `r` and `S_eff` still use the geometric top and thickness. The second call therefore appears not to change assembled coefficients; it mainly changes effective-top metadata. The intended stabilization may currently be weaker than its name suggests.

At the time of the original review, parsing assignments for `RechargeStabilizationMode` and `EffectiveTopMode` were commented out in `flow_input.h`. The subsequent implementation enables and validates both settings.

### 1.4 Partially wetted multi-node wells

Each link uses

```text
wet_length = clamp(min(screen_top,h)-screen_bottom, 0, screen_length)
cwc_eff    = cwc * max(wet_length/screen_length, 1e-12).
```

This is continuous but piecewise linear, with kinks at screen endpoints. It changes the local trace correction, trace/well coupling, well diagonal and right-hand sides, and the well-cell head recovery denominator. Because one well head connects many cells, a local wetting change has a nonlocal effect.

`dry_wells_zeroed=0` only says that no whole well is completely dry. It does not rule out large changes in partially wetted links.

### 1.5 Effective-top modes

`Nonlinear.EffectiveTopMode` accepts three values:

```ini
Nonlinear.EffectiveTopMode = off
Nonlinear.EffectiveTopMode = recharge_receivers
Nonlinear.EffectiveTopMode = all_water_table_cells
```

The effective top is an assembly elevation used to decide how long the unconfined specific-yield storage window remains active. It represents the idea that a top or water-table cell can extend vertically to the current water table instead of becoming confined as soon as the head reaches the geometric cell top.

The conductivity and storage effects must be distinguished:

* Top-layer conductivity uses the physical water column over the **geometric** cell thickness. Consequently, `r` can exceed one when `h > z_top`. This represents the additional transmissive thickness above the geometric top and applies independently of `EffectiveTopMode`.
* The effective top changes the upper logistic storage window through `assembly_z_top`. It can therefore retain a specific-yield contribution where using the geometric top would cause the cell to behave as confined storage.
* The dry, partially saturated, and fully saturated flags continue to use the geometric cell bottom and top. Effective top does not reclassify the physical cell.
* Recharge routing still occurs when effective-top mode is disabled. Recharge hysteresis and effective-top storage stabilization are separate mechanisms.

The three modes differ as follows.

#### `off`

No cell receives an effective top:

```text
assembly_z_top = geometric z_top
```

The specific-yield window is therefore controlled entirely by the geometric cell. Once head rises above the geometric top, the upper logistic factor decreases toward zero and storage approaches confined compressive storage. Top-layer transmissivity can still increase through `r > 1`.

This mode provides the least storage stabilization and most closely follows the geometric mesh. It is useful as a reference case and for determining whether effective-top storage is contributing to an oscillation. It may be more sensitive when the water table repeatedly crosses the top of a cell.

#### `recharge_receivers`

Only cells selected to receive routed recharge are assigned an effective top. For an accepted receiver, the current code uses

```text
assembly_z_top = max(h, z_top - 0.1 b),
```

where `b` is the geometric cell thickness. All other cells retain their geometric top.

This is the default and most localized stabilization. It keeps the unconfined storage treatment tied to cells through which recharge enters the saturated system, while avoiding changes to every water-table cell. It is less intrusive than `all_water_table_cells`, but the effective-top set can change if recharge moves to another receiver. Recharge hysteresis is therefore important: without a stable receiver set, storage treatment can move between cells and contribute to nonlinear oscillation.

For a receiver with head above or close to its geometric top, `assembly_z_top` follows the head or remains near the upper part of the cell. If `assembly_z_top == h`, the upper logistic storage factor is at the center of its smooth transition. The smoothing thickness `eps` controls how rapidly the specific-yield contribution changes around that point.

One subtlety is that `z_top - 0.1 b` is below the geometric top. For a receiver whose head lies lower than that elevation, the selected assembly top is therefore lower than the geometric top rather than an extension above it. This shortens the storage window and may not match the conceptual meaning of “extending the cell to the water table.” The before/after `assembly_z_top` and `S_eff` values should be checked in the detailed log, and this lower bound should be reviewed if receiver cells at low saturation are common.

#### `all_water_table_cells`

Recharge receivers receive the same treatment described above. In addition, every geometric water-table cell can receive an effective top. For a non-receiver water-table cell, the code selects

```text
assembly_z_top = h
```

provided the cell contains water above its bottom and is either partially saturated or is a top-layer water-table cell.

This is the broadest effective-top treatment. It makes unconfined storage behavior less dependent on whether a cell happens to be a recharge receiver and is useful when the water table extends through areas with little or zero recharge. It can reduce artificial confined behavior along the water-table surface, but it also makes storage coefficients head-dependent in more cells. That larger nonlinear region may improve physical consistency while increasing the work required by Picard or Anderson iteration.

#### Interaction with `RechargeStabilizationMode`

`Nonlinear.RechargeStabilizationMode = hysteresis_only` forces effective-top behavior off, regardless of the `EffectiveTopMode` value. In that case recharge-receiver hysteresis remains active, but every cell uses its geometric top for storage. To use either `recharge_receivers` or `all_water_table_cells`, the recharge stabilization mode must be

```ini
Nonlinear.RechargeStabilizationMode = effective_top
```

#### Comparison

| Effective-top mode | Cells using an effective top | Recharge still routed? | `r > 1` in top cells? | Expected nonlinear effect |
|---|---|---:|---:|---|
| `off` | None | Yes | Yes | Smallest head-dependent storage region; potentially sharper confined/unconfined transition |
| `recharge_receivers` | Current routed-recharge receiver cells | Yes | Yes | Localized storage stabilization; depends on receiver-set stability |
| `all_water_table_cells` | Recharge receivers plus all detected water-table cells | Yes | Yes | Broadest unconfined storage treatment; more head-dependent cells |

For debugging, compare the three modes with identical damping, time step, and forcing. The detailed nonlinear log should be used to compare water-table update norms, receiver changes, maximum `S_eff` changes, and mass-balance error. A mode should not be judged only by iteration count: it must also preserve the intended storage response and water balance.

## 2. Interpretation of the supplied run

| NL iteration | raw max update (m) | L2 update (m) | recharge total |
|---:|---:|---:|---:|
| 0 | 79.890 | 4003.68 | 681621.26 |
| 1 | 39.945 | 1969.77 | 681655.59 |
| 2 | 19.972 | 984.91 | 681626.67 |
| 3 | 9.986 | 494.37 | 681628.00 |
| 4 | 4.993 | 251.83 | 681637.41 |
| 5 | 2.807 | 132.59 | 681633.73 |
| 6 | 2.323 | 76.27 | 681620.98 |
| 7 | 2.916 | 48.02 | 681611.91 |
| 8 | 2.809 | 33.95 | 681630.99 |
| 9 | 1.950 | 28.94 | 681621.71 |
| 10 | 2.557 | 27.66 | 681609.24 |
| 11 | 2.813 | 24.57 | 681622.30 |

The first five maximum updates almost halve at every iteration, suggesting damping toward an initially stable raw solution (and possibly runtime `omega` near 0.5). After iteration 5, the dominant cell or active set changes and contraction stops.

The threshold is approximately

```text
0.01 + 1e-4*72.27 = 0.01723 m.
```

The plateau is 100--170 times larger, so raising the iteration limit or slightly relaxing the tolerance is unlikely to solve the underlying problem. The maximum and mean heads look smooth because one moving water-table front, receiver, or well link can oscillate without noticeably changing global statistics.

The code continues to flux recovery after exhausting the nonlinear loop. A non-converged state can therefore be accepted and passed to the next time step. A production run should instead have an explicit failure policy: reject/retry with smaller `dt`, or terminate with a clear error.

## 3. Anderson acceleration issue

The current control flow appears to prevent Anderson from accumulating usable history:

1. an Anderson attempt occurs whenever `UseAnderson` is true;
2. before `AndersonStart`, or with fewer than two history entries, it returns false;
3. every false return calls `clear_anderson_history()`;
4. one history entry is then stored;
5. the next attempt again has insufficient history and clears it.

Thus history may remain at length one indefinitely. The absence of any `Anderson acceleration accepted` line in the supplied output supports this interpretation.

There is a second bookkeeping concern. History stores `x_new` together with `f=x_new-x_old`. Standard Anderson history requires consistently paired `x_k` and fixed-point residual `H(x_k)-x_k`. If `x_new` is an accelerated accepted point, its displacement is not the underlying map residual. Anderson behavior should be verified and corrected before tuning its regularization, memory, or beta.

## 4. Water-table-focused convergence

The code already uses a global infinity norm, not the printed L2 norm:

```text
max over all DG0 cells abs(h_new-h_guess), followed by an MPI maximum.
```

The suggestion to compare water-table heads is physically sound because the strongest `r` and specific-yield nonlinearities occur there. It can stop harmless deep saturated-cell changes from controlling the strict criterion. It should not, however, use only cells classified partially saturated at the current iteration; that mask can change exactly when the water table crosses a face.

A stable convergence mask should include:

* current and previous water-table cells in every vertical column;
* cells immediately above and below them (a one-cell halo);
* top cells where `r` may exceed one;
* old and new recharge receiver cells;
* cells with partially wetted well links.

Use the union of old and new masks so a crossing cell cannot disappear from the norm. The primary metric can be the infinity norm of water-table elevation change. Its relative scale should use a physical thickness or water-table head range, not absolute elevation, because the current `max(abs(head))` depends on the vertical datum.

Water-table convergence should be combined with safeguards:

* no receiver changes for at least two iterations;
* no nonlinear cell-class changes in the active halo;
* small maximum changes in `r`, `S_eff`, and `cwc_eff`;
* acceptable global and maximum cell water-balance residuals;
* a looser full-domain head-update limit to catch deep well instability.

This is safer than either the present all-cell strict maximum or a water-table-only test with no balance check.

## 5. Prioritized improvements

### 5.1 Identify the oscillating mechanism first

Log the top update cells and state transitions described below. Then run controlled time-step-0 experiments, changing one mechanism at a time:

1. disable Anderson and use conservative Picard damping;
2. freeze recharge receivers after an initial routing pass;
3. freeze well wet fractions while retaining well coupling;
4. freeze storage while retaining `r(h)`;
5. freeze `r(h)` while retaining storage.

These are diagnostic experiments, not production formulations. The first experiment that removes the plateau identifies the dominant feedback.

### 5.2 Repair iteration bookkeeping before acceleration

Do not clear Anderson history for “too early” or “insufficient history.” Store consistent `(x_k, H(x_k)-x_k)` pairs. Restart history only for a numerical rejection or a substantial active-set change. Explicitly update and verify ghosts after accepting a nonlinear iterate.

Report both raw residual and accepted step. If an accelerated candidate is accepted, its true residual is known only after the following reassembly.

### 5.3 Adaptive stabilization

Establish a baseline with Anderson off and `omega` around 0.2--0.5. Reduce damping when the residual grows, reverses sign at the same cell, or an active set changes; increase it cautiously after several monotone contractions. Once Anderson is verified, restart it on receiver or large well-wetting transitions. A merit-function line search is safer than rejecting acceleration only by L2 step size.

### 5.4 Time-step or load continuation

The first step may be far from equilibrium. Ramp pumping/recharge, begin with smaller `dt`, and retry failed steps with reduced `dt`. Always retry from the last converged state, not the last attempted iterate.

### 5.5 Smooth and reconcile transitions

Scale storage smoothing to local thickness or expected head movement rather than one global 0.05 m. If wells are implicated, consider smooth or hysteretic screen wetting. Reconcile cell-state and recharge thresholds. Verify the intended effect of effective top and the physical meaning of `r>1`. Validate storage with a conservative stored-water derivation and a one-column benchmark.

## 6. Diagnostic output to add

Console output should contain one compact global row per nonlinear iteration. Detailed records should go to CSV or JSON-lines, using persistent global cell IDs and including time step and nonlinear iteration.

### 6.1 Iteration summary

```text
step, iter, dt,
raw_update_inf, raw_update_l2,
accepted_step_inf, accepted_step_l2,
wt_update_inf, raw_residual_ratio,
omega, accepted_method,
anderson_history_size, anderson_status, m_used,
max_alpha, accelerated_to_picard_step_ratio,
linear_iterations, linear_final_relative_residual,
head_min, head_max, head_mean,
r_min, r_max, max_delta_r,
S_min, S_max, max_delta_S,
n_dry, n_partial, n_saturated, n_class_changes,
n_receivers, n_receiver_added, n_receiver_removed, rerouted_area,
n_partial_well_links, max_delta_wet_fraction,
net_external_Q, global_mass_balance_error, max_cell_balance_error
```

For rejected Anderson steps, print a reason: too early, insufficient history, factorization failure, coefficient limit, or step-size limit.

### 6.2 Top-N update cells

Report the 10--20 largest raw updates, not only the single maximum:

```text
rank, global_cell_id, active_index, head_dof,
x, y, z, column_id, layer,
h_guess, h_raw, h_accepted, raw_dh, accepted_dh,
z_bot, z_top, distance_to_each_face,
old_class, new_class,
r_old, r_new, S_old, S_new,
old_receiver, new_receiver, routed_area, effective_top,
well_ids, wet_fraction_old, wet_fraction_new
```

Flag two-cycle behavior when update sign alternates with similar magnitude or `h_k` is close to `h_(k-2)`. Keeping several cells is important because the identity of the maximum may alternate between adjacent layers.

### 6.3 Recharge changes

Print the symmetric difference of old/new receiver global IDs. For each change record source column, old and new receiver, layer movement, routed area, `psi`, selected threshold, `r`, previous-receiver flag, and acceptance/rejection reason. Report rerouted area even when total recharge is unchanged.

### 6.4 Coefficient transitions

List cells with the largest absolute and relative changes in `r` and `S_eff`. Count cells near `z_bot` and `z_top`, at `r_min`, with `0<r<1`, and with `r>1`. Log coefficients before and after the effective-top recalculation to establish whether it changes assembly.

### 6.5 Well transitions

For partially wetted or strongly changing links, record

```text
well_id, cell_gid, h, screen_bottom, screen_top,
wet_length, wet_fraction, base_cwc, effective_cwc,
delta_effective_cwc, well_head, exchange_flow.
```

Per well report summed effective conductance, wet/total screen length, requested pumping, aquifer exchange, and imbalance. Flag screen-endpoint crossings.

### 6.6 Balance and algebraic residuals

For the recovered state report

```text
storage change + boundary outflow + pumping - recharge - stream inflow
```

as absolute and normalized errors, plus maximum cell continuity error and cell ID. For the condensed solve report `||Sx-b||`, `||b||`, relative residual, and a separate well-block residual. Inner iteration count alone is not a residual measure.

Dump full per-cell state only when the residual grows, an active set changes, a two-cycle is detected, or the iteration limit is reached.

## 7. Recommended debugging sequence

1. Print the actual runtime nonlinear controls and confirm damping and stabilization modes.
2. Run step 0 with Anderson disabled and the iteration summary enabled.
3. Find the top cells at iterations 5--11 and check update sign, layer alternation, and `h_k` versus `h_(k-2)`.
4. Correlate those cells with receiver, `r/S_eff`, and well-wetting changes.
5. Freeze one mechanism at a time to isolate the feedback.
6. Validate storage and balance in a vertical column: first without wells, then recharge, then one well.
7. Test the water-table-plus-safeguards convergence criterion.
8. Repair and verify Anderson history before re-enabling it.
9. Add explicit rejection/retry behavior at the nonlinear iteration limit.

## 8. Most likely explanation for this run

The aggregate output most strongly indicates a local active-set or coefficient oscillation after the smooth bulk error decays. Likely causes, in order to investigate rather than asserted certainty, are:

1. transfer of specific-yield storage between vertically adjacent cells;
2. local recharge receiver switching hidden by nearly constant total recharge;
3. partially wetted well-link changes;
4. a conductivity/storage kink at a cell boundary;
5. ineffective Anderson acceleration plus fixed damping that is too large for the remaining feedback.

The existing aggregate output cannot conclusively rank the first four. Top-N cell and state-transition logging is the shortest route to a definitive diagnosis.
