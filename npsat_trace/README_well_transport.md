# Well capture and instantaneous wellbore routing

This document describes two separate operations:

1. Decide whether an **aquifer particle enters a well**.
2. Route an **already captured particle through the well**, possibly returning
   it to the aquifer in another layer.

The flow-weighted capture implementation is used automatically with:

```ini
[Simulation]
TransportMethod = point
VelocityInterpolation = coarse_rt
Direction = -1
WellCaptureSubdivisions = 4
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

## 1. Flow-weighted proximity in an aquifer cell

### Reference geometry

The velocity cache's inverse map returns coordinates in `[-1,1]^3`. Capture
converts these to the unit cube using `xi = (reference + 1)/2`.

For each well, the nearest physical point on its screen segment in the current
cell is at the particle's elevation clamped to that segment's bottom and top.
The code compares the particle and that screen point using the squared distance
in all three normalized cell coordinates. A sample above or below the screen
can therefore be captured if the well's score is high enough at that distance.

Reference distances to the six faces are `xi`, `1-xi`, `eta`, `1-eta`, `zeta`,
and `1-zeta`. Both scores use reference coordinates so cell dimensions set
the distance scale.

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
well_score_j = Wj / (dw_j² + epsilon²)
face_score   = sum_f Ff / (df² + epsilon²)
epsilon      = 1e-6 reference units
```

The strongest well captures if its score exceeds the sum of the competing face
scores. Inflow faces and well-to-aquifer sources do not act as capture outlets.
Wells are sorted by internal ID to give deterministic tie ordering. Each point
is assigned to at most one well.

With one outflow face and negligible regularization, the inequality is:

```text
dw < df * sqrt(W / F)
```

This inverse-square score is a subcell spatial closure, **not** an analytical
solution for radial well flow. Volumetric budgets alone cannot uniquely locate
the physical capture streamline inside a coarse cell.

The tracer uses the loaded outlet rates directly and caches the model for each
visited well cell for the loaded flow step.

### Detecting capture during a particle step

The tracer checks the starting point and divides the proposed step into equal
pieces. It checks the endpoints in travel order, skipping samples outside the
current cell or beyond an age or water-table limit.
`WellCaptureSubdivisions` defaults to 4 and accepts positive integers.
A value of 1 checks the starting point and the step endpoint if it is eligible.
A capture region between samples can be missed.

At the first qualifying sample, aquifer travel stops and internal well routing
starts. Aquifer travel to the sample consumes its usual travel time. A step
that exits the cell without capture ends just past the face so the next cell
can apply its own field and capture model.

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

The capture model and geometry queries use already loaded local data. Existing
particle migration and global step-completion logic handle distributed tracing.

- `well_capture.h` provides local outlet scores and cell containment.
- `main_class_impl/npsat_trace_capture.impl.h` samples particle steps and builds the cell cache.
- `cached_velocity.h` provides the unclamped inverse mapping.
- `main_class_impl/npsat_trace_point.impl.h`: capture events and travel-time handling.
- `well_routing.h`: signed-flow internal outlet selection and budget check.
- `main_class_impl/npsat_trace_main.impl.h`: well-flow loading and internal routing.

## Focused verification

The lightweight numerical capture check needs only a C++11 compiler:

```sh
g++ -std=c++11 -O2 -I. tests/well_routing/capture_checks.cpp -o /tmp/capture_checks
/tmp/capture_checks
```

It checks screen-end attraction above and below the screen, competition against
face outflow, and rejection of samples outside the cell. The MPI checks are
described in `tests/well_routing/README.md`.
