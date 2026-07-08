# Well Radius, Skin Radius, and Skin Conductivity Setup

This note summarizes the role of the `rw`, `Rskin`, and `Kskin` well-input fields used by `MNWellCollection::read_well_data()` and `NPSAT_FLOW::setup_local_cell_well_link()`.

The well input file columns are:

```text
Eid,x,y,top,bottom,q_row,rw,Rskin,Kskin
```

## Where These Parameters Are Used

The fields are read from the well file in `npsat_flow/mnwells.h`:

```cpp
w.rw = std::stod(*it++);
w.Rskin = std::stod(*it++);
w.Kskin = std::stod(*it++);
```

They are later used in `npsat_flow/main_class_impl/npsat_flow_setup.impl.h` inside `setup_local_cell_well_link()` to compute the cell-well conductance, `cwc`.

The conductance is then used as:

```cpp
Qe = cwc * (hw - h_e)
```

where:

- `Qe` is the exchange flow between well and aquifer cell.
- `hw` is the well head.
- `h_e` is the aquifer/cell head.
- Positive `Qe` means injection from well to aquifer.
- Negative `Qe` means pumping from aquifer to well.

So `rw`, `Rskin`, and `Kskin` do not directly prescribe pumping. They control how strongly the aquifer cell and well exchange water.

## Conductance Formula

The code computes:

```cpp
loss_denom = 2*pi*b*KK
ln_ro_rw = log(ro/rw)
skin = (KK*b / (Kskin*bw) - 1.0) * log(Rskin_eff/rw)
D = ln_ro_rw + skin
cwc = loss_denom / D
```

where:

- `KK = sqrt(Kxx*Kyy)`, the effective horizontal hydraulic conductivity.
- `b` is the full cell thickness.
- `bw` is the screened length inside the current cell.
- `rw` is the well radius.
- `ro` is the Peaceman equivalent radius.
- `Rskin_eff` is the effective skin radius after code-side limits are applied.
- `Kskin` is the hydraulic conductivity of the skin zone.

The Peaceman radius is computed as:

```cpp
ro = 0.14 * sqrt(dx*dx + dy*dy)
```

where `dx` and `dy` are the horizontal cell dimensions.

## Units

Assuming the model uses meters and days:

| Parameter | Meaning | Units |
|---|---|---|
| `rw` | Well radius | length, usually m |
| `Rskin` | Skin radius or skin-radius fraction | dimensionless if `0 < Rskin <= 1`; length if `Rskin > 1` |
| `Kskin` | Skin-zone hydraulic conductivity | same as aquifer `K`, usually m/day |
| `cwc` | Cell-well conductance | volume/time/head, typically m2/day if head is m and flow is m3/day |
| `Qe` | Cell-well exchange flow | volume/time, typically m3/day |

## Important `Rskin` Interpretation

In the current code, `Rskin` has two possible meanings:

```cpp
0 < Rskin <= 1 : fraction of element diameter
Rskin > 1      : physical length units
```

This means:

- `Rskin = 0.2` means `20%` of the local element diameter.
- `Rskin = 0.5` means `50%` of the local element diameter.
- `Rskin = 2.0` means `2.0` model length units, usually `2 m`.
- `Rskin = 1.0` is treated as a fraction, not as `1 m`.

This is important because a value like `0.3` is not interpreted as `0.3 m`.

## Code-Side Limits on `Rskin`

The code applies several limits:

```cpp
Rskin_fraction_min = 0.1
Rskin_fraction_max = 0.5
Rskin_rw_factor    = 1.1
Rskin_ro_factor    = 0.8
```

If `Rskin` is entered as a fraction, it is first limited to:

```text
0.1 <= Rskin_fraction <= 0.5
```

Then the effective physical skin radius is forced to be at least:

```text
Rskin_eff >= 1.1 * rw
```

and no larger than:

```text
Rskin_eff <= 0.8 * ro
```

Since:

```text
ro = 0.14 * element_diameter
```

the upper limit is:

```text
Rskin_eff <= 0.8 * 0.14 * element_diameter
Rskin_eff <= 0.112 * element_diameter
```

Therefore, even though fractional `Rskin` values up to `0.5` are accepted initially, values larger than about `0.112` of the element diameter will usually be capped later by the `0.8*ro` limit.

## Meaning of the Warnings

### Fraction Smaller Than Minimum

Warning:

```cpp
Warning: well <id> has Rskin fraction <value> smaller than 0.1. Using 0.1.
```

This means the input value satisfies:

```text
0 < Rskin < 0.1
```

The code treated `Rskin` as a fraction of the element diameter and replaced it with:

```text
Rskin = 0.1
```

Example:

```text
Rskin = 0.05
```

means `5%` of the element diameter, but the code uses `10%` instead.

### Fraction Larger Than Maximum

Warning:

```cpp
Warning: well <id> has Rskin fraction <value> larger than 0.5. Using 0.5.
```

This means the input value satisfies:

```text
0.5 < Rskin <= 1.0
```

The code treated `Rskin` as a fraction of the element diameter and replaced it with:

```text
Rskin = 0.5
```

Example:

```text
Rskin = 0.75
```

means `75%` of the element diameter, but the code uses `50%` instead.

## How to Avoid the Warnings

To avoid the two fraction warnings, use:

```text
0.1 <= Rskin <= 0.5
```

when using fraction mode.

To avoid the later effective-radius cap as well, use a value close to:

```text
0.10 <= Rskin <= 0.112
```

This is because the code caps the physical skin radius at `0.8*ro`, which is approximately `0.112` of the element diameter.

The most conservative default is:

```text
Rskin = 0.1
```

in fraction mode.

## Reasonable Ranges

### `rw`: Well Radius

Use the actual physical borehole or screen radius.

Typical values:

| Well type | Approximate `rw` |
|---|---:|
| Small monitoring well | `0.025-0.075 m` |
| Typical production well | `0.1-0.3 m` |
| Large municipal/agricultural well | `0.3-0.6 m` |

Avoid:

- `rw <= 0`
- `rw >= ro`
- unrealistically large `rw` relative to cell size

### `Rskin`: Skin Radius

Recommended approach:

```text
Rskin = 0.1
```

This uses fraction mode and sets the skin radius to about `10%` of the local element diameter.

If using physical length mode:

```text
Rskin > 1
```

then choose a value that satisfies:

```text
1.1*rw < Rskin < 0.8*ro
```

Be careful: values less than or equal to `1` are not interpreted as physical lengths.

### `Kskin`: Skin Hydraulic Conductivity

`Kskin` should have the same units as aquifer hydraulic conductivity.

Its effect comes through:

```cpp
KK*b / (Kskin*bw)
```

If:

```text
Kskin = KK*b/bw
```

then the skin term is approximately zero.

Practical ranges:

| Condition | Suggested `Kskin` |
|---|---:|
| No known skin effect | near `KK*b/bw` |
| Mild clogging/damage | `0.1-0.5 * K` |
| Severe clogging/damage | `0.001-0.1 * K` |
| Stimulated or high-conductivity completion | `2-100 * K` |

Avoid:

- `Kskin <= 0`
- values many orders of magnitude smaller than `K` unless severe clogging is intended
- values many orders of magnitude larger than `K` unless strong stimulation is intended

## Practical Setup Recommendation

A conservative starting point per well is:

```text
rw     = actual well/screen radius
Rskin  = 0.1
Kskin  = local aquifer K, or KK*b/bw if accounting for partial screen length
```

Then inspect output quantities such as:

- `cwc`
- `Qe`
- total well exchange
- wellbore segment flows

If the well exchanges too little water for a given head difference, increase `Kskin` or reduce skin resistance.

If the well exchanges too much water, reduce `Kskin` to represent clogging or well loss.

## Key Takeaways

- `rw` is a physical well radius.
- `Kskin` is a hydraulic conductivity and must use the same time/length units as aquifer `K`.
- `Rskin` is dimensionless only when `0 < Rskin <= 1`.
- `Rskin > 1` is interpreted as a physical radius in model length units.
- `Rskin = 0.1` is a good conservative starting value in the current code.
- The warnings mean the input fraction was outside the code's allowed range and was clipped.
- To avoid the fraction warnings, use `0.1 <= Rskin <= 0.5`.
- To avoid most effective-radius clipping too, use `Rskin` close to `0.1`.
