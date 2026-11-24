Well-routing regression checks
==============================

Build separately from the application (deal.II with MPI is required):

```sh
cmake -S tests/well_routing -B build-well-tests -DCMAKE_BUILD_TYPE=Debug
cmake --build build-well-tests -j2
mpirun -n 1 build-well-tests/well_routing_tests
mpirun -n 2 build-well-tests/well_routing_tests
```

The harness compiles the actual tracer with private access exposed only in a
generated build-directory copy. It constructs a four-cell vertical well and
uses the real ParticleHandler to test MPI migration of captured particles.
No simulation input files are needed. Checks cover zero-exchange throughflow,
20/80 partial discharge, reversed crossflow, pumping discharge, and no backward
capture at extracting well segments. The outlet helper also checks converging
and diverging axial flows, stagnant segments, and inconsistent budgets.

The one- and two-rank runs should report identical outlet counts. Internal
routing uses a perfectly mixed segment approximation with reproducible spatial
sampling. It consumes no particle travel time. Existing proximity capture and
coarse-RT aquifer velocities are not changed or calibrated by these tests.
