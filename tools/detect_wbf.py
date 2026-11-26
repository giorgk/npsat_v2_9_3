"""Flag likely wellbore flow from simplified NPSAT streamline coordinates."""

from math import isfinite

import numpy as np
import pandas as pd


def detect_wbf(
    streamlines_csv,
    well_geometry_csv,
    *,
    well_radius=20.0,
    xy_tol=0.5,
    min_dz=1.0,
    max_xy_to_z=0.05,
):
    """Return one row per (Eid, Sid) with a boolean wbf flag.

    The streamline CSV must contain Eid, Sid, x, y, z, with samples in
    traversal order within each streamline. The well CSV must contain
    x, y, top, bottom for vertical wells. All distances use the coordinate
    units of the CSVs.

    A segment qualifies if its horizontal displacement is at most xy_tol
    and max_xy_to_z times its vertical displacement, both endpoints are
    within well_radius of the same well axis, and at least min_dz of the
    segment's vertical extent overlaps that well's screen. Any well can
    qualify, regardless of the streamline's seed Eid. The flag is inferred
    from geometry and depends on which points the simplifier retained.
    """
    for name, value in (
        ("well_radius", well_radius),
        ("xy_tol", xy_tol),
        ("min_dz", min_dz),
        ("max_xy_to_z", max_xy_to_z),
    ):
        if not isfinite(value) or value < 0 or (name == "min_dz" and value == 0):
            raise ValueError(f"Invalid {name}: {value}")

    keys = ["Eid", "Sid"]
    coords = ["x", "y", "z"]
    points = pd.read_csv(streamlines_csv, usecols=keys + coords)
    well_cols = ["x", "y", "top", "bottom"]
    wells = pd.read_csv(well_geometry_csv, usecols=well_cols)[well_cols]
    if points[keys].isna().any().any():
        raise ValueError("Streamline Eid and Sid must be present on every row")
    xyz = points[coords].to_numpy(dtype=float)
    geometry = wells.to_numpy(dtype=float)
    if not np.isfinite(xyz).all() or not np.isfinite(geometry).all():
        raise ValueError("Streamline coordinates and well geometry must be finite")
    if (geometry[:, 2] < geometry[:, 3]).any():
        raise ValueError("Every well screen must have top >= bottom")

    points[coords] = xyz
    start = points.groupby(keys, sort=False)[coords].shift().to_numpy()
    dxy = np.hypot(xyz[:, 0] - start[:, 0], xyz[:, 1] - start[:, 1])
    dz = np.abs(xyz[:, 2] - start[:, 2])
    candidates = np.flatnonzero((dz >= min_dz) & (dxy <= xy_tol) & (dxy <= max_xy_to_z * dz))
    a, b = start[candidates], xyz[candidates]
    z_low = np.minimum(a[:, 2], b[:, 2])
    z_high = np.maximum(a[:, 2], b[:, 2])

    matched = np.zeros(len(candidates), dtype=bool)
    for wx, wy, top, bottom in geometry:
        screen_overlap = np.minimum(z_high, top) - np.maximum(z_low, bottom)
        matched |= (
            (screen_overlap >= min_dz)
            & (np.hypot(a[:, 0] - wx, a[:, 1] - wy) <= well_radius)
            & (np.hypot(b[:, 0] - wx, b[:, 1] - wy) <= well_radius)
        )

    flags = np.zeros(len(points), dtype=bool)
    flags[candidates[matched]] = True
    return points[keys].assign(wbf=flags).groupby(keys, sort=False, as_index=False)["wbf"].any()
