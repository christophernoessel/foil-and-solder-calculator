# Stained Glass Foil & Solder Cost Calculator

A command-line tool that reads an SVG stained glass pattern, measures all the piece edges, classifies them as interior joints or exterior boundary, and estimates material costs for copper foil tape and 60/40 solder.

## What it does

You give it an SVG file of your pattern and one real-world dimension (e.g. "the finished piece is 18 inches wide"). The tool:

1. Parses every closed shape in the SVG (paths, polygons, rectangles, circles, ellipses — including inside transformed groups).
2. Identifies **interior edges** (shared by two adjacent pieces) and **exterior edges** (the outside boundary hidden under the frame).
3. Calculates total foil needed as the sum of every piece's full perimeter, since each piece gets foiled regardless of whether its edge is interior or exterior.
4. Calculates solder needed based on interior joint length only, soldered on both front and back.
5. Adds a 5% waste factor and reports costs based on current supplier pricing.

## Usage

```
python stained_glass_calculator.py                  # opens a file picker dialog
python stained_glass_calculator.py my_pattern.svg   # use a specific file
```

The tool will prompt you for:

- **Dimension and direction** — one known measurement of the finished piece (horizontal or vertical), in inches or centimeters.
- **Foil width and backing** — common choices are 7/32" copper-backed (the default), 3/16", or 1/4", with copper, black, or silver backing.

### Example output

```
  Pieces:                  13
  Total perimeter (all):   246.8 in
  Interior joints:         35.3 in
  Exterior edges:          176.3 in

--- Foil ---
  Selected foil:           EDCO 7/32" copper-backed
  Total foil needed:       259.2 in  (incl. 5% waste)
  Rolls needed:            1
  FOIL COST:               $15.95

--- Solder (60/40, .125" dia.) ---
  Interior joints:         35.3 in
  Soldered length (×2):    6.2 ft  (front + back, incl. 5% waste)
  Estimated solder:        0.39 lbs
  SOLDER COST:             $9.07

  TOTAL MATERIALS:         $25.02
```

## Requirements

- Python 3.8+
- No third-party packages — uses only the standard library (`xml.etree`, `math`, `re`, `tkinter`).
- `tkinter` is only needed for the file picker dialog. If it's not installed, pass the SVG path as a command-line argument instead.

## SVG requirements

The tool works best when each glass piece is a **separate closed shape** in the SVG — either a `<path>` ending with `Z`, a `<polygon>`, `<rect>`, `<circle>`, or `<ellipse>`. Adjacent pieces must share exact coordinates along their common edges for the tool to correctly identify interior joints.

### Supported SVG features

- `<path>` with all standard commands: M, L, H, V, C, S, Q, T, A, Z (absolute and relative)
- `<rect>`, `<circle>`, `<ellipse>`, `<polygon>`, `<polyline>`, `<line>`
- `transform` attributes: `translate`, `scale`, `rotate`, `matrix`, `skewX`, `skewY`
- Nested `<g>` groups with inherited transforms
- `viewBox` and `width`/`height` for scaling

### Not supported

- `<use>` / `<defs>` references
- CSS-driven geometry or styling that affects shape
- `<clipPath>` or `<mask>` elements
- Shapes defined only by stroke (no fill) — the tool looks at geometry, not rendering

## How edge classification works

Each segment of each closed path gets a canonical key based on its endpoints (and control points for curves). If two different pieces produce the same key, that edge is interior. If only one piece owns it, it's exterior.

This means:

- If two pieces share an edge but their coordinates are slightly off (even by a fraction of an SVG unit), the tool will classify both copies as exterior — overcounting foil and undercounting solder.
- The tool runs a cross-check: `total_perimeter ≈ exterior + 2 × interior`. If your numbers don't add up, it likely means some edges aren't aligning.

## Pricing

Prices are hardcoded from these suppliers as of February 2026:

- **Copper foil**: [Stained Glass For Less](https://www.stainedglassforless.com/copper-foil-tape/) — EDCO and Studio Pro foil, 36-yard rolls, $10.95–$21.95 depending on width and backing.
- **Solder**: [Pacwest Supply](https://pacwestsales.com/products/amerway-60-40-solder-for-stained-glass-125-dia-5-pack) — Amerway 60/40, .125" diameter, 5 × 1 lb spools for $117.50 ($23.50/lb).

Edit the `FOIL_CATALOG` and `SOLDER_PRICE_PER_LB` constants at the top of the script to update prices.

## Solder estimate accuracy

The solder calculation uses a rule of thumb: roughly 1 lb of 60/40 wire solder covers about 16 linear feet of finished bead (both sides combined). Actual consumption varies significantly with bead width, technique, and how much you touch up. Treat this as a ballpark — it could be off by 30% or more in either direction.

## License

MIT
