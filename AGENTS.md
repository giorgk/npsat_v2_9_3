# NPSAT development guide

## Project layout

- `npsat_v2.cpp` builds the groundwater-flow executable, `npsat_v2`.
- `npsat_trace.cpp` builds the particle-tracing executable, `npsat_trace`.
- `npsat_flow/` and `npsat_trace/` contain the corresponding implementation headers and sources.
- The project uses deal.II 9.3+ with MPI and Trilinos support.

## Build and run

Configure an out-of-source build and point CMake at the deal.II installation when needed:

```powershell
cmake -S . -B build -DDEAL_II_DIR=<path-to-deal.II>
cmake --build build --config Release
```

The generated executables are `npsat_v2` and `npsat_trace`. Run MPI programs with the launcher appropriate to the installed MPI implementation.

## Change guidelines

- Preserve the existing C++11 compatibility and deal.II coding patterns.
- Keep MPI ownership, ghost-vector synchronization, and distributed-triangulation changes explicit and local.
- Treat flow outputs as the interface consumed by tracing; update both sides when an output format, prefix, or mesh/velocity mapping changes.
- Do not commit build directories, generated simulation outputs, or IDE-local files.

## Verification

At minimum, configure and build both CMake targets after changing shared code. For solver or tracer behavior changes, run a representative small MPI case and check generated outputs for the affected step.
