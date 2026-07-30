# Review of `NPSAT_TRACE::run_trajectory_atlas`

This document reviews the trajectory-atlas tracing driver
(`npsat_trace/main_class_impl/npsat_trace_atlas.impl.h`) together with the
data structures it depends on (`npsat_trace/traj_atlas.h`). It describes the
overall control flow, the mathematics behind the atlas construction and query,
comments on **deal.II 9.3.2 / gcc‑5 / C++11** compatibility, and closes with
suggestions for improvement.

> **Goal of the tracing code.** Rather than linking a receptor to a *single*
> point at the source, the atlas method links a receptor to a **distribution of
> source points** ("representative source area"). Each backward-traced *branch*
> carries a `weight` (its share of the receptor's flow), an `age`, and an
> `aquifer_length`. The age/length pair is consumed by a later step that
> simulates **longitudinal dispersion**, so preserving them accurately per
> branch is a first-class requirement.

---

## 1. High-level control flow

The method traces particles **backward** in time (from receptor toward source)
and is asserted to run only when `direction < 0`.

```
run_trajectory_atlas()
├── load_triangulation(); setup_triangulation_helpers(); setup_system()
├── read delta_times[]            # transient step lengths
├── ParticleReader over the seed file (rank 0)
└── for each chunk of seeds:                         # "iteration"
     ├── rank 0 builds one AtlasPacket per seed, routes it to the
     │   owning rank by bounding box (atlas_owner_for_position)
     ├── exchange_atlas_packets()  -> local `active` set
     ├── flow_step = last step;  time_pass = 0
     └── while (packets remain):                      # transient "time pass"
          ├── load_data_step(flow_step); reset dt_remaining = delta_times[step]
          ├── while (global unfinished > 0):          # "exchange epoch"
          │    ├── drain local_queue (DFS via back()/pop_back()):
          │    │     • dt_remaining <= dt_eps      -> completed (step_complete)
          │    │     • age >= max_age              -> write "3 0" (truncated)
          │    │     • generation >= cap           -> write "3 0" (truncated)
          │    │     • cell not local              -> forward to owner / write
          │    │     • else: build cache+atlas, atlas.query(pos, dt_remaining)
          │    │            split into child branches (see §3)
          │    ├── exchange_atlas_packets(send) -> received; active.swap
          │    └── stop when global unfinished == 0, or n_max_proc_exchanges hit
          ├── active.swap(completed)                  # carry step_complete on
          └── flow_step = (flow_step - 1) mod Nsteps  # walk backward in time
```

Three nested loops therefore exist: **chunks of seeds → transient time passes →
MPI exchange epochs**, with the innermost work being a depth-first drain of the
local packet queue.

### Termination / boundedness
- **Exchange epoch** is capped by `n_max_proc_exchanges`; leftover packets are
  flushed to output as truncated.
- **Generation** (`child.generation = parent.generation + 1`) increments on
  every cell crossing and is capped by `n_max_streamline_steps`.
- **Age** is capped by `max_age` (when `>= 0`).
- **Local integration** during atlas *construction* is capped by
  `maximum_local_steps`.

These four caps make the whole procedure provably terminating even in the
presence of stagnation or numerical loops.

---

## 2. Atlas construction (`build_cell_atlas`)

Each locally owned cell builds a **cell-local transfer atlas** for a given
`flow_step`, cached and reused (`get_or_build_cell_atlas`) keyed on
`cell->user_index()` with a validity flag and the flow step it was built for.

### 2.1 Subface flows and sign convention

For each of the 6 hex faces (and up to 4 refined subfaces per face → 24
canonical "slots"), the outward normal flux is

$$
Q_f \;=\; A_f \, (\mathbf{v}\cdot\mathbf{n})_f ,
$$

with the convention that **positive = forward outflow** from the aquifer cell.
Thus:

- forward **outflow**: $Q_f > \varepsilon_Q$ (a backward trajectory *starts* here),
- forward **inflow**: $Q_f < -\varepsilon_Q$ (a backward trajectory *ends* here),

where $\varepsilon_Q$ = `flow_tolerance`. Wells intersecting the cell are added
as extra terminals with an analogous `outward_flow` (direction-adjusted:
`Qe * direction`).

### 2.2 Seeding backward trajectories

Every forward-outflow subface is seeded with $n_q \times n_q$ launch points on
a bilinear parameterization of the (quad) subface:

$$
\mathbf{x}(u,v)=\sum_{k=0}^{3} N_k(u,v)\,\mathbf{p}_k,\qquad
N=\big[(1-u)(1-v),\;u(1-v),\;(1-u)v,\;uv\big],
$$

with $u,v=(i+\tfrac12)/n_q$ (cell-centered midpoint rule). Each launch point
carries an equal share of the face flow:

$$
q_{\text{traj}} \;=\; \frac{Q_f}{n_q^2}.
$$

Points are nudged toward the cell center by $10^{-8}$ to stay strictly interior.

### 2.3 Local backward integration

From each launch point the code integrates the (porosity-scaled) velocity
field **backward**:

$$
\mathbf{v}_{\text{eff}}=\frac{\mathbf{v}}{n_e},\qquad
\hat{\mathbf{d}}=\frac{\mathbf{v}_{\text{eff}}}{\lVert\mathbf{v}_{\text{eff}}\rVert},\qquad
\Delta s=\max\!\Big(\frac{w(\hat{\mathbf{d}})}{20},\,10^{-10}\Big),\qquad
\Delta t=\frac{\Delta s}{\lVert\mathbf{v}_{\text{eff}}\rVert},
$$

where $n_e$ is porosity and $w(\hat{\mathbf d})$ is the directional bounding-box
width (an adaptive step ≈ 1/20 of the cell extent along the travel direction).
The update is explicit Euler in space:

$$
\mathbf{x}_{k+1}=\mathbf{x}_k-\Delta s\,\hat{\mathbf{d}} .
$$

Each sample stores cumulative **backward time** (age proxy) and **aquifer arc
length**:

$$
\tau_{k+1}=\tau_k+\Delta t,\qquad \ell_{k+1}=\ell_k+\Delta s .
$$

Integration stops on one of:
- **well capture** (`well_bore_flow_trace`) → `AtlasTerminal::well`,
- **stagnation** ($\lVert\mathbf v_{\text{eff}}\rVert \le$ `stagnant_velocity_threshold`) → `stagnant`,
- **face exit** (`find_hex_exit`) → matched to the nearest subface slot; kept only if that subface is a **forward inflow**, else `unresolved`,
- **step budget exhausted** → `unresolved`.

Each finished trajectory is resampled to `stored_samples_per_trajectory`
uniformly-spaced points in $\tau$ (`resample_atlas_trajectory`), keeping the
$(\mathbf x,\tau,\ell)$ triples monotone.

### 2.4 Transfer table and mass balance

`finalize()` aggregates trajectories into a downstream→upstream transfer table.
For an origin (subface or well) with total forward outflow $Q_{\text{out}}(o)$,
the fraction routed to terminal $t$ is

$$
f(o\!\to\!t)=\frac{\sum_{a:\,o\to t} q_{\text{traj},a}}{Q_{\text{out}}(o)} .
$$

Two diagnostics are recorded: the cell **mass-balance residual**
$\sum Q_{\text{out}} - \sum Q_{\text{in}}$ and the **maximum origin relative
error** $\max_o |{\textstyle\sum_a q_{\text{traj},a}} - Q_{\text{out}}(o)|/Q_{\text{out}}(o)$,
verifying that the seeded trajectories reproduce the face flow.

---

## 3. Atlas query and branching (`CellTrajectoryAtlas::query`)

Given an interior packet position $\mathbf x_0$ and the time budget
`available_time` = `dt_remaining`, the query builds a **flow-weighted,
distance-weighted** estimate of which terminals the point drains to.

1. Gather all trajectory samples, compute an **anisotropic squared distance**
   using per-axis physical scales $\sigma_d$ (cell extents):

$$
D^2(\mathbf x_0,\mathbf x_s)=\sum_{d} \Big(\frac{x_{0,d}-x_{s,d}}{\sigma_d}\Big)^2 .
$$

2. Keep the `maximum_query_samples` nearest samples. Each contributes a weight
   that combines a **volumetric (residence) weight** and an **IDW kernel**:

$$
w = \underbrace{q_{\text{traj}}\cdot \Delta\tau_s}_{\text{volume}}\;\cdot\;
\underbrace{\big(D^2+\epsilon^2\big)^{-p/2}}_{\text{kernel}},
$$

where $\Delta\tau_s$ is the sample's control time (half the neighbor time
interval), $p$ = `kernel_power`, $\epsilon$ = `kernel_epsilon`.

3. Accumulate per **terminal** $t$ the weighted branch fraction and weighted
   conditional quantities:

$$
f_t=\frac{\sum_{s\in t} w_s}{\sum_s w_s},\qquad
\overline{\Delta t}_t=\frac{\sum_{s\in t} w_s\,\Delta t_s}{\sum_{s\in t} w_s},\qquad
\overline{\Delta \ell}_t=\frac{\sum_{s\in t} w_s\,\Delta \ell_s}{\sum_{s\in t} w_s},
$$

along with a weighted **advanced position** $\overline{\mathbf x}_t$. The
advance uses $\Delta t_s=\min(\text{available\_time},\,\text{remaining\_time}_s)$,
so a branch either reaches its terminal (`reaches_terminal = true` when the
weighted remaining time fits inside the budget) or stops at an interior point to
be continued in the next transient step.

4. Branches below `minimum_branch_fraction` are dropped (their fraction is
   reported as discarded), and the survivors are **renormalized to sum to 1** so
   splitting stays conservative.

### 3.1 Packet splitting in the driver

Back in `run_trajectory_atlas`, the query result is optionally truncated to
`maximum_branches_per_packet` (largest fractions kept, renormalized again), and
each branch spawns a child:

$$
w_{\text{child}}=w_{\text{parent}}\cdot f_t,\qquad
\text{age}_{\text{child}}=\text{age}_{\text{parent}}+\overline{\Delta t}_t,\qquad
\ell_{\text{child}}=\ell_{\text{parent}}+\overline{\Delta \ell}_t .
$$

A runtime assertion checks $\big|\sum_t w_{\text{child}} - w_{\text{parent}}\big|
\le 10^{-10}\max(|w_{\text{parent}}|,1)$ — i.e. **weight is conserved** across the
split. A child that:
- does **not** reach a terminal → becomes `step_complete` at the interior point,
- reaches an interior **subface** → migrates to the neighbor cell (local queue
  or MPI send), nudged toward the neighbor center,
- reaches a **boundary subface / well / stagnant / unresolved** terminal → is
  written to the per-rank output file with `weight, age, length, position,
  terminal_kind, terminal_id`.

This is exactly the mechanism that yields **multiple weighted source points per
receptor**, each with its own accumulated age and length for downstream
dispersion modeling.

---

## 4. Compatibility (deal.II 9.3.2 / gcc-5 / C++11)

Nothing in the reviewed code requires a standard newer than C++11, and the
deal.II surface used is present in 9.3.x. Specific points:

- **C++ features:** `enum class`, `static_assert`, lambdas, `std::array`,
  `<cstdint>` fixed-width types, in-class member initializers, and brace-init
  (`metric_scale{{1.0,1.0,1.0}}`) are all C++11. No `auto` return deduction,
  structured bindings, `if constexpr`, `std::optional`, or generic lambdas are
  used. ✔
- **deal.II API:** `DoFHandler::active_cell_iterator`, `cell->point_inside`,
  `cell->neighbor_child_on_subface`, `cell->user_index`, `GeometryInfo`,
  `Utilities::MPI::sum`, `Utilities::int_to_string`, `Point`/`Tensor` are all
  available in 9.3.x. ✔
- **MPI:** raw `MPI_Alltoall` / `MPI_Alltoallv` with `MPI_BYTE` over a
  fixed-layout POD (`AtlasPacketWire`) is portable. Note this relies on all
  ranks using the same struct padding (true for a homogeneous build), which is
  the standard assumption here. ✔
- One nit: `MPI_Alltoallv` is passed `0` (not `nullptr`) for empty buffers —
  fine in C++11.

No compatibility blockers were found.

---

## 5. Correctness observations

These are behaviors worth being aware of; none is an outright defect given the
current modeling intent, and **no code was changed**.

1. **Linear cell search is the dominant cost.**
   `find_owned_cell_for_atlas_packet` scans *every* active cell with
   `point_inside` for *every* packet dequeued, and `atlas_owner_for_position`
   scans every rank's bounding boxes. With many packets × branches × cells this
   is `O(N_cells)` per packet and likely the runtime bottleneck.

2. **Mass reallocation on branch pruning.** Both `minimum_branch_fraction`
   (in `query`) and `maximum_branches_per_packet` (in the driver) drop small
   branches and *renormalize survivors to 1*. This conserves total weight but
   silently **reassigns the pruned mass** to the retained terminals, slightly
   biasing the source-area distribution toward dominant paths. The discarded
   fraction is computed but not emitted to the output file.

3. **Truncated packets and genuine terminals share output codes.** Age-cap,
   generation-cap, non-local-unresolvable, invalid-query, and
   proc-exchange-overflow packets all write `terminal_kind = 3` (unresolved),
   indistinguishable in the output from an integration `unresolved`. A
   downstream consumer cannot tell "hit the wall of a cap" from "numerically
   lost the path."

4. **Weight-conservation assertion is absolute-toleranced.** The check uses
   `1e-10 * max(|weight|, 1)`. For deep generations `weight ≪ 1`, so the scale
   floors at 1 and the tolerance is effectively absolute `1e-10`. Summing many
   renormalized fractions could in principle drift near this bound; it is an
   `AssertThrow`, so a borderline case aborts the run rather than warning.

5. **`Δτ` control-time weighting near the origin.** `sample_control_time`
   returns `0.5*(τ₁−τ₀)` for the first sample; with the first sample at `τ=0`
   this correctly gives it half-interval influence. Single-sample trajectories
   get a control time of `1.0` (a unit fallback) — reasonable but arbitrary.

6. **Nearest-subface slot matching on exit** picks the subface with the closest
   *center* on the exit face. For strongly refined neighbor faces this could
   occasionally mis-assign the slot; the forward-inflow check afterward guards
   against routing into a non-inflow face (marking it `unresolved` instead).

7. **Atlas cache is single-flow-step per cell.** `all_cell_atlas_valid` +
   `all_cell_atlas_flow_step` store one step per cell slot. Because each
   transient *time pass* holds `flow_step` constant this is fine within a pass,
   but atlases are rebuilt every pass a cell is revisited under a different
   step.

---

## 6. Suggestions for improvement

Ordered roughly by expected impact. (Documentation only — implement separately.)

### Performance
1. **Replace the linear cell search** with a `GridTools::Cache` /
   `find_active_cell_around_point`, or exploit locality: after a branch crosses
   a face you already know the neighbor cell (it is computed for migration), so
   pass it directly instead of re-locating by coordinates. This is the single
   biggest win.
2. **Cache atlases per `(cell, flow_step)`** if memory allows, so revisiting a
   time step in a later pass reuses the built transfer table instead of
   rebuilding trajectories.
3. **Prune query candidates spatially.** `query` currently materializes *all*
   samples of *all* trajectories and sorts them. A small per-cell KD-tree (the
   project already embeds `nanoflann`) over sample positions would cut this to
   the `maximum_query_samples` neighborhood.

### Fidelity of the source-area / dispersion outputs
4. **Emit distinct terminal codes** for age-cap, generation-cap,
   exchange-overflow, and integration-unresolved, and **write the discarded
   fraction** per packet. Since the product feeds a dispersion model, knowing
   *why* mass left the domain (real boundary vs. truncation) materially changes
   interpretation.
5. **Record path variability, not just cumulative mean age/length.** Because
   longitudinal dispersion depends on the *spread* of travel times, consider
   carrying a second moment (variance of $\tau$ and $\ell$) through the
   accumulators in `query`, not only the flow-weighted mean. The atlas already
   has per-sample $\tau,\ell$, so accumulating $\sum w\,\tau^2$ is cheap and
   gives the dispersion step a within-branch variance for free.
6. **Reconsider renormalizing pruned mass.** For representative source areas it
   may be preferable to *carry* the discarded fraction as an explicit
   "unattributed" sink rather than inflating retained branches, so the source
   map integrates to the true captured fraction.

### Robustness
7. **Guard the branch-conservation assertion.** Convert the `AssertThrow` on
   weight conservation into a tolerance that scales with branch count, or a
   diagnostic warning with renormalization, so a benign floating-point drift in
   a long run does not abort a multi-hour MPI job.
8. **Higher-order local integration.** Atlas construction uses explicit Euler
   with a $\Delta s \approx w/20$ step. An RK2/midpoint step (or curvature-based
   step control) would reduce the age/length bias that directly propagates into
   the dispersion estimate, at modest cost since it is a one-time per-cell build.
9. **Validate `maximum_branches_per_packet ≥ 1` at the driver** (config already
   rejects `0`, but a local assertion documents the dependency that a `resize`
   to 0 would break the weight-conservation invariant).
10. **Surface build-time diagnostics.** `mass_balance_error` and
    `maximum_origin_relative_error` are computed and only written to an optional
    debug stream. Aggregating a per-rank max to `pcout` per chunk would make
    silent atlas-quality degradation visible during production runs.

---

## 7. Summary

`run_trajectory_atlas` implements a backward, transient, MPI-distributed
particle tracer built on precomputed **cell-local transfer atlases**. Its design
aligns well with the stated goal: packet **branching with conserved weights**
produces a *distribution* of source points per receptor, and each branch
propagates an **age** and **aquifer length** suitable for a later longitudinal
dispersion step. The mathematics — flow-weighted, IDW-kernel interior queries
over residence-time-weighted trajectory samples, with per-origin mass-balance
checks — is coherent and self-validating.

The code is C++11 / gcc-5 / deal.II 9.3.x compatible with no blockers. The main
opportunities are **performance** (eliminate the per-packet linear cell search;
reuse neighbor cells; KD-tree the query) and **output fidelity for the
dispersion consumer** (distinguish truncation terminals, carry travel-time
variance, and treat pruned mass explicitly rather than renormalizing it away).
