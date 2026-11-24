# Well capture and instantaneous wellbore routing

This document describes two separate operations:

1. Decide whether an **aquifer particle enters a well**.
2. Route an **already captured particle through the well**, possibly returning
   it to the aquifer in another layer.

The flux-calibrated capture implementation is used automatically with:

```ini
[Simulation]
TransportMethod = point
VelocityInterpolation = coarse_rt
Direction = -1
```

Use `Direction = 1` for forward tracking. The flow run must have exported coarse
velocities (`Output.Save_coarse_velocities = 1`). All existing mesh, velocity,
cell/well-link, well-flow, and water-table inputs are still required. No new
input files, particle-file columns, or flow-model outputs are introduced.

Other velocity schemes and trajectory-atlas capture retain the legacy proximity
selection. The internal routing routine is shared. The four legacy parameters
`WellCaptureDistance`, `WellCaptureCellFraction`, `WellInfluenceQScale`, and
`WellInfluenceMaxCellFraction` do **not** control the new point/coarse-RT capture.

## Available flows and sign conventions

The flow output supplies the aquifer/well exchange `Qe` and axial flows at the
bottom and top of each segment, `Qwbf_bot` and `Qwbf_top`.

- Positive `Qe`: well to aquifer.
- Negative `Qe`: aquifer to well.
- Positive axial flow: upward, at either end of the segment.
- All rates have units of physical volume per model time.

Segment continuity is:

```text
Qwbf_top = Qwbf_bot - Qe
```

The existing input reader multiplies all three well flows by tracking direction.
Capture multiplies face-normal Darcy fluxes by that same direction when forming
face rates. Consequently the same formulas apply in both directions; well flows
must not be reversed a second time.

A net pumping well can still have locally discharging segments. Backward
particles can enter those segments. An idle pump can also have interlayer
crossflow: a zero net pumping rate does not mean zero segment exchange.

## 1. Flux-calibrated proximity in an aquifer cell

### Reference geometry

The velocity cache's inverse map returns coordinates in `[-1,1]^3`. Capture
converts these to the unit cube using `xi = (reference + 1)/2`.

Both the particle and well axis use the same inverse map. In the supported
extruded hexahedra, the horizontal reference coordinates of the vertical well
are constant. The squared distance to the axis is:

```text
dw² = (xi_particle - xi_well)² + (eta_particle - eta_well)²
```

Capture is eligible only inside the physical elevation interval of this cell's
screen intersection. Distance is to the axis, not to the segment midpoint.
Screen limits are not replaced with reference-height limits: sloping layers can
give different physical elevations at equal reference height.

Reference distances to the six faces are `xi`, `1-xi`, `eta`, `1-eta`, `zeta`,
and `1-zeta`. This normalizes displacement by cell geometry. It is intentionally
a reference-space metric, not an isotropic physical-distance metric.

### Competing outlet scores

Define, in the active tracking direction:

```text
Wj = max(-Qe_j, 0)             # aquifer-to-well outlet rate
Ff = max(Qface_f, 0)           # outward aquifer face rate
```

Face rates are normal Darcy flux times physical area, summed over the face's
canonical subfaces. For coarse RT, the face value is constant across those
slots. Porosity does not enter a comparison of volumetric rates.

For eligible well j and particle reference position:

```text
well_score_j = lambda_j * Wj / (dw_j² + epsilon²)
face_score   = sum_f Ff / (df² + epsilon²)
epsilon      = 1e-6 reference units
```

The strongest well captures if its score exceeds the sum of the competing face
scores. Inflow faces and well-to-aquifer sources do not act as capture outlets.
Wells are sorted by internal ID to give deterministic tie ordering. Each point
is assigned to at most one well.

With one outflow face and negligible regularization, the inequality is:

```text
dw < df * sqrt(lambda * W / F)
```

This inverse-square score is a subcell spatial closure, **not** an analytical
solution for radial well flow. Volumetric budgets alone cannot uniquely locate
the physical capture streamline inside a coarse cell.

### Local calibration of lambda

`lambda = 1` is flow-weighted but is not generally flux-calibrated. To estimate
lambda, the implementation builds a small set of representative local paths:

1. Use 4 by 4 quadrature points on each inflow face. Weight them using physical
   surface Jacobians, normalized so their weights sum to the available face
   inflow rate. Nudge their positions inward by `1e-8` reference units.
2. Include positive `Qe` as well-source inflow. Since its subcell distribution
   is not supplied, use 4 screen-height by 4 azimuth samples, uniformly weighted
   by the source rate, at a `1e-6` reference offset from the well axis. This is
   an explicit release-distribution assumption, not additional measured data.
3. Trace with the frozen coarse-RT field. Use midpoint integration, a maximum
   distance of one thirty-second of the directional cell width, and a local
   midpoint-velocity-change limit. Stop at a cell exit, stagnation, negligible
   motion, a known water table, or 256 steps. Seeds above a known water table
   are recorded as unresolved inflow, not silently redistributed. No trajectory
   leaves this cell for calibration.
4. Apply the same screened, segment-intersection capture rule used for actual
   particles. Count each path's represented flow only at its first capture.
5. Fit coefficients jointly by bounded coordinate bisection in log(lambda),
   between -24 and +24, at most 14 iterations per well and three sweeps. Retain
   the coefficient vector minimizing summed squared relative well-flow errors.

The convergence tolerance is the larger of 1% of incoming flow and the largest
quadrature path's flow weight. Thus a small target below the quadrature resolution
cannot be claimed to have a precise calibrated fraction. A single well or a
cell already within tolerance avoids unnecessary fitting.

This is a bounded deterministic approximation. Multiple-well spatial competition
can prevent simultaneous exact matches. The implementation retains the best
fit and reports an unresolved target rather than asserting exact conservation.

Calibration also checks whether uncaptured paths exited the cell. Uncaptured
paths stopped by stagnation or the integration limit contribute an unresolved
flow diagnostic; excessive unresolved flow prevents a successful-calibration
classification.

### Incomplete budgets and diagnostics

The available cell budget compares:

```text
inflow  = incoming faces + positive Qe sources
outflow = outgoing faces + negative Qe sink magnitudes
```

No storage, recharge, or other source term is invented. When the relative
imbalance exceeds 1%, or there is no represented inflow, calibration is skipped
and `lambda = 1` is retained. This fallback is **uncalibrated flow weighting**.
Transient storage, omitted source terms, or inconsistent coarse face fluxes can
therefore cause a fallback even with otherwise valid input files.

Warnings identify the cell and report budget residual, capture residual, and
unresolved trajectory flow. At most eight warnings are printed per rank per
loaded flow step. Missing linked well-flow records, nonfinite rates, invalid
screen geometry, or failed coordinate inversions cause explicit errors.

Calibration is lazy: only visited well-containing cells with attracting
segments are calibrated. The result is cached for that loaded flow step and
discarded when new flow data are loaded.

### Detecting capture during a particle step

Capture checks the path segment, not just its endpoints. A path can therefore
enter a capture region even if neither endpoint lies inside it.

A physical step is mapped into reference-space chords. Midpoint mapping error
above `1e-7` triggers subdivision, up to 16 levels; unresolved mapping error is
reported. Each chord is searched left to right using conservative bounds on
well and face scores. Search stops at `1e-7` reference resolution or depth 24,
with additional closest-axis and screen-boundary checks at that resolution.
These are numerical tolerances, not exact geometric intersection guarantees.

For point/coarse-RT tracing, aquifer steps stop just past the current cell face
using a small geometric nudge. The next cell's interior is handled with its own
field and capture model. Capture searches are clipped to the current cell and
to earlier age/water-table events. The particle advances only to the detected
entry point before internal well routing starts. That aquifer travel consumes
its usual travel time; well routing does not.

No independent random capture trial is performed at each aquifer substep.
Smaller substeps therefore do not create extra probabilistic chances of capture.
Aquifer integration accuracy, finite quadrature, and mapping tolerances still
affect the result; this is not a claim of exact step-size invariance.

The calibrated percentages concern **represented water flow**, not arbitrary
particle counts. User seeds need appropriate flow weights or a representative
distribution before particle counts can be interpreted as discharge fractions.

## 2. Instantaneous internal well routing

Once captured, proximity and the sign of `Qe` no longer determine whether the
particle stays in the well. The same well is followed using its segment flows.

The outgoing rates of a segment are:

```text
up      = max( Qwbf_top, 0)
down    = max(-Qwbf_bot, 0)
aquifer = max( Qe,       0)
```

Divide each by their sum to obtain outlet probabilities. One positive outlet
gives a deterministic route. Several outlets use a reproducible sample seeded
from particle position, segment-bottom elevation, and well ID. This represents
perfect mixing within the segment. Co-located particles obtain the same route;
the sample is independent of MPI processing order, but is not specified to be
bitwise identical across different standard-library implementations.

Examples (all in the active tracking direction):

| Bottom | Top | Qe | Result |
|---:|---:|---:|---|
| 100 | 100 | 0 | Continue upward through a zero-exchange segment |
| 100 | 80 | 20 | 80% upward, 20% aquifer release |
| -40 | 60 | -100 | 40% downward, 60% upward |
| 40 | -60 | 100 | Both axial inflows discharge to the aquifer |

The continuity check uses absolute tolerance `1e-12` plus `1e-10` times the
largest segment-flow magnitude. Invalid budgets and broken connections terminate
with `er_well_mass_balance` (13). The flow exporter assumes a closed bottom;
only a selected upward outlet at the upper screen end is treated as pumped
discharge, `er_well_captured` (12).

All vertical hops happen without advancing particle age or consuming remaining
time. No artificial dt is added: such a dt would make age depend on the number
of well segments. A wholly stagnant segment retains the captured well ID and
waits out the current flow step; that waiting, unlike movement, advances age.

The existing routing geometry preserves x/y and updates z across segment ends;
it does not snap a captured particle to the physical well axis. Individual
internal hops are not written as streamline samples. This is a layer-transfer
approximation, not resolved borehole hydrodynamics or residence-time modeling.

One particle property, `pWellId`, preserves captured-well membership when the
existing ParticleHandler machinery transfers a particle to another processor.
No internal stage history is recorded. A route started at the end of an aquifer
time step is still completed without adding time, subject to existing termination
and processor-exchange limits.

## Communication and implementation locations

The capture model, calibration trajectories, coefficient fitting, and geometry
queries perform **no MPI communication**. They use already loaded local data.
The application's existing particle migration and global step-completion logic
remain responsible for distributed tracing. No new collective, message exchange,
or cross-rank calibration is added.

- `well_capture.h`: local outlet scores, segment search, coefficient fitting.
- `main_class_impl/npsat_trace_capture.impl.h`: mapping, local flow budgets,
  quadrature trajectories, lazy cell cache, diagnostics.
- `cached_velocity.h`: access to the existing unclamped inverse and forward map.
- `main_class_impl/npsat_trace_point.impl.h`: capture events and travel-time handling.
- `well_routing.h`: signed-flow internal outlet selection and budget check.
- `main_class_impl/npsat_trace_main.impl.h`: well-flow loading and internal routing.

## Focused verification

The lightweight numerical capture check needs only a C++11 compiler:

```sh
g++ -std=c++11 -O2 -I. tests/well_routing/capture_checks.cpp -o /tmp/capture_checks
/tmp/capture_checks
```

It checks single- and multiple-well calibration, an interior capture missed by
both endpoints, screen gating, and incomplete-budget fallback. It is not an
end-to-end validation of a particular flow simulation. The earlier internal
routing MPI checks are described in `tests/well_routing/README.md`.
