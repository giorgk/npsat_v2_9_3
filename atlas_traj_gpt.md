# Trajectory-atlas review

## Scope and conclusion

This review covers `NPSAT_TRACE<3>::run_trajectory_atlas()`, its atlas builder,
and the local `CellTrajectoryAtlas` query.  The method implements backward,
transient, MPI-distributed branching from receptor seeds toward upstream
boundary or well terminals.  Its purpose is well aligned with representative
source-area tracing: instead of retaining one deterministic path from each
receptor seed, it assigns conservative weights to multiple upstream branches.

The implementation is broadly coherent and uses C++11-era language and
standard-library facilities.  It should be source-compatible with GCC 5 in
C++11 mode.  Its deal.II usage is consistent with the project's stated 9.3
baseline, but an actual deal.II 9.3.2 build is still needed to validate the
installed MPI, Trilinos, and deal.II configuration.

The principal scientific limitation is that each query merges all nearby
sampled trajectories having the same terminal into one averaged branch.  This
can collapse distinct source locations and mixes their conditional travel age
and aquifer length.  That is contrary to the stated goal when the later
longitudinal-dispersion model needs a representative distribution of source
points, ages, and lengths.

## Overall execution flow

1. `run()` selects the atlas method only when `TransportMethod::trajectory_atlas`
   is configured.  `run_trajectory_atlas()` requires a negative tracking
   direction, so it is backward tracing only.
2. The method loads the distributed triangulation, initializes mesh helpers and
   the RT0 system, reads the list of transient step durations, and opens the
   receptor-seed reader on rank zero.
3. Rank zero reads seeds in chunks.  Each seed becomes one `AtlasPacket` with
   weight \(w=1\), age \(a=0\), aquifer length \(\ell=0\), and stable receptor
   and seed identifiers.  Packets are sent to the rank whose owned-region
   bounding box contains the seed.
4. Each transient pass loads the RT0/well data for the current flow step.  The
   first pass is the final flow step and subsequent passes move backward
   cyclically through the time steps.
5. For each active packet, the owning rank locates its locally owned cell,
   lazily builds that cell's atlas for the current flow step, and queries the
   atlas with the packet's position and remaining duration.
6. The query may return several branches.  The driver creates one child packet
   per retained branch, advances its age and aquifer length, and either:
   - completes it for this transient step when it remains inside the cell;
   - moves it across an internal subface, locally or by MPI exchange; or
   - writes it as a boundary, well, stagnant, or unresolved terminal.
7. After each MPI exchange epoch, globally received packets become the next
   active set.  When no packet remains in the current step, the step-complete
   packets are advanced to the preceding flow step.  The chunk ends when no
   globally active packet remains.

`exchange_atlas_packets()` uses a count exchange followed by `MPI_Alltoallv`
with a fixed wire representation.  This is a sensible design: C++ object
layout and padding never cross rank boundaries, and cells retain ownership of
their cached atlases.

## Cell-atlas construction

`build_cell_atlas()` creates a separate atlas for each locally owned cell and
flow step.

### Boundary representation

For each physical outer face or refined child subface, the builder stores its
center, area \(A_j\), and forward outward normal velocity \(u_{n,j}\).  The
forward subface flow is

$$
Q_j = A_j u_{n,j}.
$$

Thus \(Q_j>0\) is forward outflow from the cell, and \(Q_j<0\) is forward
inflow.  The builder also loads cell-well exchange rates, applying the tracing
direction to make their signs comparable to this convention.

Only forward-outflow subfaces are used as local atlas origins.  Each origin is
sampled on an \(n_q\times n_q\) tensor grid; every sampled local trajectory is
assigned represented flow

$$
q_k = \frac{Q_j}{n_q^2}.
$$

This makes the collection of trajectories from an origin sum to that origin's
forward flow, before numerical/terminal failures.

### Backward local paths

Each quadrature point is nudged just inside the cell and integrated opposite to
the pore velocity.  The cache first reconstructs velocity, then the code uses

$$
\mathbf v_p = \frac{\mathbf v}{n_e},
\qquad
\Delta s = \max\!\left(\frac{W(\hat{\mathbf v}_p)}{20},10^{-10}\right),
\qquad
\Delta t = \frac{\Delta s}{\lVert\mathbf v_p\rVert},
$$

where \(n_e\) is effective porosity and \(W\) is the cached directional cell
width.  It records monotonically increasing backward elapsed time and aquifer
arc length.  A trajectory ends at a forward-inflow subface, a well event, a
stagnant point, or an unresolved event.  It is then time-resampled to the
configured stored-sample count.

The atlas finalization computes cell flow diagnostics

$$
Q_{\rm in}=\sum_j\max(-Q_j,0)+\sum_w\max(-Q_w,0),
\qquad
Q_{\rm out}=\sum_j\max(Q_j,0)+\sum_w\max(Q_w,0),
$$

and reports \(Q_{\rm out}-Q_{\rm in}\).  It also checks whether the total
represented trajectory flow for every outflow origin matches that origin's
flow.  These are diagnostics only; they do not reject a poor atlas.

## Interior query, branching, and transport quantities

At a packet position \(\mathbf x\), all stored trajectory samples are ranked
by anisotropically scaled squared distance,

$$
d^2(\mathbf x,\mathbf x_i)=
\sum_{r=1}^{3}\left(\frac{x_r-x_{i,r}}{s_r}\right)^2,
$$

where the scales \(s_r\) are the current cell extents.  The query keeps at
most `MaximumQuerySamples` nearby samples.  Sample \(i\) receives kernel
weight

$$
\omega_i = q_i\,\Delta t_i^{\rm control}
\left(d_i^2+\epsilon^2\right)^{-p/2}.
$$

The represented flow \(q_i\) and control-time factor approximate a local
water-volume measure; the inverse-distance term localizes the query.

Samples are then grouped **by terminal identity**.  For terminal \(T\), the
raw branch fraction is

$$
f_T = \frac{\sum_{i\in T}\omega_i}{\sum_i\omega_i}.
$$

Fractions below `MinimumBranchFraction` are dropped; the remaining fractions
are normalized to one.  The driver additionally retains at most
`MaximumBranchesPerPacket` branches and renormalizes again.  Therefore, for a
parent packet with weight \(w\), retained children satisfy

$$
w_b = w f_b,
\qquad \sum_b w_b=w.
$$

For each retained branch, the code adds the query's weighted conditional
advance to the persistent packet state:

$$
a_{b}^{\rm new}=a^{\rm old}+\Delta t_b,
\qquad
\ell_{b}^{\rm new}=\ell^{\rm old}+\Delta\ell_b.
$$

`aquifer_length` deliberately excludes instantaneous well-bore routing, which
is appropriate if the later dispersion calculation is intended to model
aquifer transport only.  The output records receptor ID, seed ID, lineage,
weight, age, aquifer length, position, and terminal identifier.

## Findings

### What is sound

- Backward-only enforcement matches receptor-to-source tracing.
- Packet lineage, weight, age, and aquifer length persist across local queues
  and MPI serialization.
- Renormalization and the explicit weight-conservation assertion prevent loss
  of represented mass when branch pruning is applied.
- Caching the atlas per owned cell and transient step avoids rebuilding it for
  every packet.
- The step-completion behavior is correct in intent: if available transient
  time is insufficient to reach the local terminal, the branch advances to an
  interior position and waits for the next flow field.
- Flow and origin-representation diagnostics provide valuable checks on local
  transfer quality.

### Important limitations and risks

1. **Terminal aggregation loses source-area detail.** `query()` accumulates
   every nearby trajectory with the same `AtlasTerminal` into one branch and
   averages its position, elapsed time, and length.  Multiple distinct paths
   to the same boundary subface or well become one synthetic path.  The
   resulting averaged position need not lie on any physical trajectory.  More
   importantly, a later longitudinal-dispersion calculation receives a mean
   instead of the age/length distribution it needs.

2. **Averaging makes the reach decision inconsistent for mixed paths.** A
   grouped branch declares `reaches_terminal` from the *mean* remaining time.
   Some contributing paths can have reached the terminal while others have not.
   Advancing their averaged position and treating the group as one crossing
   can bias both transient timing and source allocation.

3. **Well terminal attribution is approximate.** On a well event, the atlas
   selects the nearest screened well to the current point rather than having
   `well_bore_flow_trace()` return the captured well ID.  If capture/routing
   changes position or there are nearby screened wells, the recorded terminal
   can be wrong.  The `well_id=0` fallback also conflates a missing match with
   a legitimate identifier of zero.

4. **Aquifer time/length omit the final well-capture displacement.** When the
   well rule fires, the path terminates before adding a segment to the
   well-event position.  This is reasonable only when capture is explicitly
   treated as instantaneous at the current aquifer point.  It should be stated
   and used consistently in any dispersion model.

5. **Unresolved, age-limit, iteration-limit, and exchange-limit outcomes all
   use terminal kind `3`.** That enum value means `unresolved`, so downstream
   analysis cannot distinguish a numerical failure from a deliberate maximum
   age or iteration termination.

6. **Diagnostics do not enforce quality.** `BalanceRelativeTolerance` is read
   but is not used to warn, reject, or adapt an atlas.  The builder can retain
   trajectories ending in unresolved/stagnant terminals while only logging a
   diagnostic if a debug stream is supplied (the driver passes a closed stream).

7. **Repeated transient cycling has no independent horizon.** If no terminal,
   maximum age, or generation limit is reached, the driver cycles through the
   supplied flow steps indefinitely.  The generation limit is a count of atlas
   queries, not a transparent physical-time horizon.

8. **Cell lookup is linear in locally owned active cells for every packet.**
   `find_owned_cell_for_atlas_packet()` may dominate cost for large partitions.

## Compatibility review

### C++11 and GCC 5

The atlas code uses C++11-supported features: `auto`, lambdas without generic
parameters, `enum class`, `static_assert`, fixed-width integer types,
`std::array`, `std::vector`, `std::map`, `std::unordered_map`, and C++11
default member initializers.  It does not rely on C++14/17 language features
such as generic lambdas, structured bindings, `std::optional`, `if constexpr`,
or filesystem support.  The project already compiles with `-std=c++11`.

The fixed packet wire format is explicitly appropriate for an older compiler:
it serializes scalar fields rather than MPI-transmitting a C++ object.  Use a
compiler/MPI combination with compatible `std::uint64_t` and IEEE-754 `double`
representations across participating ranks, as is normal for a homogeneous
HPC run.

### deal.II 9.3.2

The code confines itself to established distributed-triangulation and
`DoFHandler` operations used elsewhere in this project: active-cell iterators,
local-ownership checks, `point_inside`, face/neighbor access, user indices,
and `Utilities::MPI`.  No newer deal.II syntax is apparent in the atlas path.
The project requests deal.II 9.3.0 in CMake, so 9.3.2 satisfies the configured
minimum.

This is a static compatibility review, not a substitute for compilation:
verify with the exact 9.3.2 headers and MPI/Trilinos build.  In particular,
exercise refined interfaces because `neighbor_child_on_subface()` and the
canonical subface-slot mapping must agree for coarse-to-refined transfers.

## Recommended improvements

1. Preserve a distribution, not one branch per terminal.  First group by
   individual sampled trajectory (or clusters defined by terminal **and**
   position/time/length bins).  Split the packet among the retained samples,
   then prune with a documented error tolerance.  This directly produces
   multiple weighted source points and retains a distribution of \((a,\ell)\).
2. If grouping is required for performance, keep at least weighted moments per
   terminal: \(E[a]\), \(E[\ell]\), variances, covariance, and an effective
   sample count.  A dispersion stage should consume those moments or the
   individual branches rather than only a mean path.
3. Make terminal events explicit: add distinct output/status codes for
   boundary, well, stagnant, unresolved, maximum age, maximum generation, and
   maximum MPI-exchange termination.  Include discarded and retained fractions
   in output or diagnostics so source-area mass accounting is auditable.
4. Return the captured well ID and event position from `well_bore_flow_trace()`
   instead of inferring the nearest well.  Define whether the capture segment
   contributes to aquifer age and length, then implement that definition in
   both point and atlas paths.
5. Use `BalanceRelativeTolerance`: emit a rank-safe diagnostic and optionally
   reject/rebuild an atlas when flow residual or origin representation error is
   too large.  Track the total unresolved/stagnant represented fraction.
6. Add an explicit physical maximum backward time or maximum number of complete
   forcing cycles.  Report it separately from the per-query generation cap.
7. Replace the per-packet linear cell scan with a point-location strategy based
   on the particle handler, cached last cell, or a deal.II grid-search utility
   that is available in the pinned 9.3.2 environment.
8. Add focused regression cases: one-to-many source splitting, two distinct
   trajectories to the same terminal, coarse/refined face crossing, MPI
   partition crossing, well capture, a stagnant region, and a transient step
   that ends before a local terminal.  Verify weight conservation and compare
   the weighted distributions of age and length, not only their means.
