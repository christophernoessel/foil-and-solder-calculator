#!/usr/bin/env python3
"""
Stained Glass Foil & Solder Cost Calculator

Parses an SVG stained glass pattern, classifies edges as interior joints
or exterior boundary, and estimates material costs for copper foil and solder.

Usage:
    python stained_glass_calculator.py              # opens file dialog
    python stained_glass_calculator.py pattern.svg  # uses specified file

Requirements: Python 3.8+, tkinter (for file dialog if no path given)
"""

import sys
import os
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ---------------------------------------------------------------------------
# Pricing data (fetched Feb 2026)
# ---------------------------------------------------------------------------

# EDCO copper foil – 36-yard (1296-inch) rolls
# stainedglassforless.com/copper-foil-tape/
FOIL_CATALOG = {
    # (width_inches, backing): (price_usd, brand, roll_length_inches)
    (3/16, "copper"):  (13.95, "EDCO", 1296),
    (7/32, "copper"):  (15.95, "EDCO", 1296),
    (1/4,  "copper"):  (17.95, "EDCO", 1296),
    (3/16, "black"):   (17.95, "EDCO", 1296),
    (7/32, "black"):   (19.95, "EDCO", 1296),
    (1/4,  "black"):   (21.95, "EDCO", 1296),
    (3/16, "silver"):  (17.95, "EDCO", 1296),
    (7/32, "silver"):  (19.95, "EDCO", 1296),
    (1/4,  "silver"):  (21.95, "EDCO", 1296),
    # Studio Pro – 36-yard rolls
    (7/32, "copper-sp"):  (10.95, "Studio Pro", 1296),
    (7/32, "black-sp"):   (12.95, "Studio Pro", 1296),
    (7/32, "silver-sp"):  (12.95, "Studio Pro", 1296),
}

# Amerway 60/40 solder, .125" dia, 5×1 lb spools = $117.50
# pacwestsales.com – sale price as of Feb 2026
SOLDER_PRICE_PER_LB = 117.50 / 5  # $23.50/lb
SOLDER_LBS_PER_FOOT_OF_SEAM = 1.0 / 16.0  # ~1 lb covers 16 ft of seam (both sides)
# This means for each foot of *interior joint*, soldered on front AND back,
# we consume about 1/8 lb.  (16 ft = 8 ft of joint × 2 sides)

WASTE_FACTOR = 1.05  # 5% extra for mistakes / cleanup

# ---------------------------------------------------------------------------
# 2-D point helpers
# ---------------------------------------------------------------------------

def pt_eq(a, b, tol=0.5):
    """Check if two points are close enough to be considered equal."""
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

# ---------------------------------------------------------------------------
# SVG path "d" attribute parser
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A single segment of a path: line, cubic bezier, quadratic bezier, or arc."""
    kind: str  # 'L', 'C', 'Q', 'A'
    pts: list  # control/end points depending on kind
    # For 'L': [start, end]
    # For 'C': [start, cp1, cp2, end]
    # For 'Q': [start, cp1, end]
    # For 'A': stored differently (see below)
    arc_params: Optional[dict] = None

    def length(self, n_steps=64) -> float:
        if self.kind == 'L':
            return dist(self.pts[0], self.pts[1])
        elif self.kind == 'C':
            return _cubic_length(self.pts, n_steps)
        elif self.kind == 'Q':
            return _quad_length(self.pts, n_steps)
        elif self.kind == 'A':
            return _arc_length(self)
        return 0.0

    def start(self):
        return self.pts[0]

    def end(self):
        return self.pts[-1]

    def reversed_seg(self):
        if self.kind == 'L':
            return Segment('L', [self.pts[1], self.pts[0]])
        elif self.kind == 'C':
            return Segment('C', [self.pts[3], self.pts[2], self.pts[1], self.pts[0]])
        elif self.kind == 'Q':
            return Segment('Q', [self.pts[2], self.pts[1], self.pts[0]])
        elif self.kind == 'A':
            p = self.arc_params
            return Segment('A', [self.pts[-1], self.pts[0]],
                           arc_params={**p, 'sweep': 1 - p['sweep']})
        return self

    def canon_key(self, tol=0.5) -> tuple:
        """Return a hashable canonical key for edge matching.
        We round coordinates and pick the lexicographically smaller direction."""
        def rnd(p):
            return (round(p[0] / tol) * tol, round(p[1] / tol) * tol)

        if self.kind == 'L':
            a, b = rnd(self.pts[0]), rnd(self.pts[1])
            return ('L', min(a, b), max(a, b))
        elif self.kind == 'C':
            fwd = tuple(rnd(p) for p in self.pts)
            rev = tuple(rnd(p) for p in reversed(self.pts))
            return ('C',) + min(fwd, rev)
        elif self.kind == 'Q':
            fwd = tuple(rnd(p) for p in self.pts)
            rev = tuple(rnd(p) for p in reversed(self.pts))
            return ('Q',) + min(fwd, rev)
        elif self.kind == 'A':
            # For arcs, use sampled midpoint plus endpoints
            a, b = rnd(self.pts[0]), rnd(self.pts[-1])
            mid = rnd(self._arc_midpoint())
            fwd = (a, mid, b)
            rev = (b, mid, a)
            return ('A',) + min(fwd, rev)
        return ('?',)

    def _arc_midpoint(self):
        """Sample the midpoint of an arc segment."""
        if self.arc_params and 'cx' in self.arc_params:
            p = self.arc_params
            mid_angle = (p['theta1'] + p['theta1'] + p['dtheta']) / 2.0
            mid_angle_rad = math.radians(mid_angle)
            cos_phi = math.cos(math.radians(p['phi']))
            sin_phi = math.sin(math.radians(p['phi']))
            x = p['cx'] + p['rx'] * math.cos(mid_angle_rad) * cos_phi - p['ry'] * math.sin(mid_angle_rad) * sin_phi
            y = p['cy'] + p['rx'] * math.cos(mid_angle_rad) * sin_phi + p['ry'] * math.sin(mid_angle_rad) * cos_phi
            return (x, y)
        return lerp(self.pts[0], self.pts[-1], 0.5)


def _cubic_length(pts, n=64):
    """Approximate cubic bezier length by sampling."""
    total = 0.0
    prev = pts[0]
    for i in range(1, n + 1):
        t = i / n
        t1 = 1 - t
        x = t1**3 * pts[0][0] + 3*t1**2*t * pts[1][0] + 3*t1*t**2 * pts[2][0] + t**3 * pts[3][0]
        y = t1**3 * pts[0][1] + 3*t1**2*t * pts[1][1] + 3*t1*t**2 * pts[2][1] + t**3 * pts[3][1]
        total += dist(prev, (x, y))
        prev = (x, y)
    return total


def _quad_length(pts, n=64):
    """Approximate quadratic bezier length by sampling."""
    total = 0.0
    prev = pts[0]
    for i in range(1, n + 1):
        t = i / n
        t1 = 1 - t
        x = t1**2 * pts[0][0] + 2*t1*t * pts[1][0] + t**2 * pts[2][0]
        y = t1**2 * pts[0][1] + 2*t1*t * pts[1][1] + t**2 * pts[2][1]
        total += dist(prev, (x, y))
        prev = (x, y)
    return total


def _arc_length(seg, n=64):
    """Approximate arc length by converting to center parameterization and sampling."""
    p = seg.arc_params
    if not p or 'cx' not in p:
        # Fallback: straight line
        return dist(seg.pts[0], seg.pts[-1])

    total = 0.0
    prev = seg.pts[0]
    cos_phi = math.cos(math.radians(p['phi']))
    sin_phi = math.sin(math.radians(p['phi']))

    for i in range(1, n + 1):
        t = i / n
        angle = math.radians(p['theta1'] + p['dtheta'] * t)
        x = p['cx'] + p['rx'] * math.cos(angle) * cos_phi - p['ry'] * math.sin(angle) * sin_phi
        y = p['cy'] + p['rx'] * math.cos(angle) * sin_phi + p['ry'] * math.sin(angle) * cos_phi
        total += dist(prev, (x, y))
        prev = (x, y)
    return total


# ---------------------------------------------------------------------------
# SVG arc endpoint-to-center conversion (per SVG spec)
# ---------------------------------------------------------------------------

def _arc_endpoint_to_center(x1, y1, rx, ry, phi_deg, large_arc, sweep, x2, y2):
    """Convert SVG arc endpoint parameterization to center parameterization."""
    phi = math.radians(phi_deg)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    dx2 = (x1 - x2) / 2
    dy2 = (y1 - y2) / 2
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    rx = abs(rx)
    ry = abs(ry)

    # Ensure radii are large enough
    lam = (x1p**2) / (rx**2) + (y1p**2) / (ry**2)
    if lam > 1:
        s = math.sqrt(lam)
        rx *= s
        ry *= s

    num = max(0, rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2)
    den = rx**2 * y1p**2 + ry**2 * x1p**2
    sq = math.sqrt(num / den) if den > 0 else 0
    if large_arc == sweep:
        sq = -sq

    cxp = sq * rx * y1p / ry
    cyp = -sq * ry * x1p / rx

    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2

    def angle(ux, uy, vx, vy):
        n = math.sqrt(ux**2 + uy**2) * math.sqrt(vx**2 + vy**2)
        if n == 0:
            return 0
        c = (ux * vx + uy * vy) / n
        c = max(-1, min(1, c))
        a = math.degrees(math.acos(c))
        if ux * vy - uy * vx < 0:
            a = -a
        return a

    theta1 = angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = angle((x1p - cxp) / rx, (y1p - cyp) / ry,
                   (-x1p - cxp) / rx, (-y1p - cyp) / ry)

    if sweep == 0 and dtheta > 0:
        dtheta -= 360
    elif sweep == 1 and dtheta < 0:
        dtheta += 360

    return {'cx': cx, 'cy': cy, 'rx': rx, 'ry': ry,
            'phi': phi_deg, 'theta1': theta1, 'dtheta': dtheta,
            'large_arc': large_arc, 'sweep': sweep}


# ---------------------------------------------------------------------------
# Parse SVG "d" attribute into segments
# ---------------------------------------------------------------------------

_CMD_RE = re.compile(r'([MmZzLlHhVvCcSsQqTtAa])')
_NUM_RE = re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?')


def _tokenize_d(d_str: str) -> list:
    """Split a path 'd' attribute into command tokens."""
    # Insert separator before each command letter
    parts = _CMD_RE.split(d_str)
    tokens = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _CMD_RE.fullmatch(part):
            tokens.append(part)
        else:
            nums = _NUM_RE.findall(part)
            tokens.extend(float(n) for n in nums)
    return tokens


def parse_path_d(d_str: str) -> List[List[Segment]]:
    """Parse an SVG path 'd' attribute into a list of sub-paths,
    where each sub-path is a list of Segments."""
    tokens = _tokenize_d(d_str)
    subpaths = []
    current_segments = []
    cx, cy = 0.0, 0.0  # current point
    sx, sy = 0.0, 0.0  # sub-path start
    last_cp = None  # last control point (for S/T)
    last_cmd = None

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if isinstance(tok, str):
            cmd = tok
            i += 1
        else:
            # Implicit repeat of last command (L after M, etc.)
            if last_cmd in ('M', 'm'):
                cmd = 'L' if last_cmd == 'M' else 'l'
            else:
                cmd = last_cmd
                if cmd is None:
                    break

        def grab(n):
            nonlocal i
            vals = []
            for _ in range(n):
                if i < len(tokens) and isinstance(tokens[i], (int, float)):
                    vals.append(float(tokens[i]))
                    i += 1
                else:
                    vals.append(0.0)
            return vals

        if cmd == 'M':
            if current_segments:
                subpaths.append(current_segments)
                current_segments = []
            x, y = grab(2)
            cx, cy = x, y
            sx, sy = x, y
            last_cp = None
            last_cmd = cmd
            continue

        elif cmd == 'm':
            if current_segments:
                subpaths.append(current_segments)
                current_segments = []
            dx, dy = grab(2)
            cx, cy = cx + dx, cy + dy
            sx, sy = cx, cy
            last_cp = None
            last_cmd = cmd
            continue

        elif cmd in ('Z', 'z'):
            if not pt_eq((cx, cy), (sx, sy), tol=0.01):
                current_segments.append(Segment('L', [(cx, cy), (sx, sy)]))
            cx, cy = sx, sy
            if current_segments:
                subpaths.append(current_segments)
                current_segments = []
            last_cp = None
            last_cmd = cmd
            continue

        elif cmd == 'L':
            x, y = grab(2)
            current_segments.append(Segment('L', [(cx, cy), (x, y)]))
            cx, cy = x, y

        elif cmd == 'l':
            dx, dy = grab(2)
            nx, ny = cx + dx, cy + dy
            current_segments.append(Segment('L', [(cx, cy), (nx, ny)]))
            cx, cy = nx, ny

        elif cmd == 'H':
            x, = grab(1)
            current_segments.append(Segment('L', [(cx, cy), (x, cy)]))
            cx = x

        elif cmd == 'h':
            dx, = grab(1)
            nx = cx + dx
            current_segments.append(Segment('L', [(cx, cy), (nx, cy)]))
            cx = nx

        elif cmd == 'V':
            y, = grab(1)
            current_segments.append(Segment('L', [(cx, cy), (cx, y)]))
            cy = y

        elif cmd == 'v':
            dy, = grab(1)
            ny = cy + dy
            current_segments.append(Segment('L', [(cx, cy), (cx, ny)]))
            cy = ny

        elif cmd == 'C':
            x1, y1, x2, y2, x, y = grab(6)
            current_segments.append(Segment('C', [(cx, cy), (x1, y1), (x2, y2), (x, y)]))
            last_cp = (x2, y2)
            cx, cy = x, y

        elif cmd == 'c':
            dx1, dy1, dx2, dy2, dx, dy = grab(6)
            cp1 = (cx + dx1, cy + dy1)
            cp2 = (cx + dx2, cy + dy2)
            end = (cx + dx, cy + dy)
            current_segments.append(Segment('C', [(cx, cy), cp1, cp2, end]))
            last_cp = cp2
            cx, cy = end

        elif cmd == 'S':
            x2, y2, x, y = grab(4)
            if last_cp and last_cmd in ('C', 'c', 'S', 's'):
                x1 = 2 * cx - last_cp[0]
                y1 = 2 * cy - last_cp[1]
            else:
                x1, y1 = cx, cy
            current_segments.append(Segment('C', [(cx, cy), (x1, y1), (x2, y2), (x, y)]))
            last_cp = (x2, y2)
            cx, cy = x, y

        elif cmd == 's':
            dx2, dy2, dx, dy = grab(4)
            if last_cp and last_cmd in ('C', 'c', 'S', 's'):
                x1 = 2 * cx - last_cp[0]
                y1 = 2 * cy - last_cp[1]
            else:
                x1, y1 = cx, cy
            cp2 = (cx + dx2, cy + dy2)
            end = (cx + dx, cy + dy)
            current_segments.append(Segment('C', [(cx, cy), (x1, y1), cp2, end]))
            last_cp = cp2
            cx, cy = end

        elif cmd == 'Q':
            x1, y1, x, y = grab(4)
            current_segments.append(Segment('Q', [(cx, cy), (x1, y1), (x, y)]))
            last_cp = (x1, y1)
            cx, cy = x, y

        elif cmd == 'q':
            dx1, dy1, dx, dy = grab(4)
            cp = (cx + dx1, cy + dy1)
            end = (cx + dx, cy + dy)
            current_segments.append(Segment('Q', [(cx, cy), cp, end]))
            last_cp = cp
            cx, cy = end

        elif cmd == 'T':
            x, y = grab(2)
            if last_cp and last_cmd in ('Q', 'q', 'T', 't'):
                cpx = 2 * cx - last_cp[0]
                cpy = 2 * cy - last_cp[1]
            else:
                cpx, cpy = cx, cy
            current_segments.append(Segment('Q', [(cx, cy), (cpx, cpy), (x, y)]))
            last_cp = (cpx, cpy)
            cx, cy = x, y

        elif cmd == 't':
            dx, dy = grab(2)
            if last_cp and last_cmd in ('Q', 'q', 'T', 't'):
                cpx = 2 * cx - last_cp[0]
                cpy = 2 * cy - last_cp[1]
            else:
                cpx, cpy = cx, cy
            end = (cx + dx, cy + dy)
            current_segments.append(Segment('Q', [(cx, cy), (cpx, cpy), end]))
            last_cp = (cpx, cpy)
            cx, cy = end

        elif cmd == 'A':
            rx, ry, phi, large_arc, sweep, x, y = grab(7)
            large_arc = int(round(large_arc))
            sweep = int(round(sweep))
            arc_p = _arc_endpoint_to_center(cx, cy, rx, ry, phi, large_arc, sweep, x, y)
            seg = Segment('A', [(cx, cy), (x, y)], arc_params=arc_p)
            current_segments.append(seg)
            cx, cy = x, y

        elif cmd == 'a':
            rx, ry, phi, large_arc, sweep, dx, dy = grab(7)
            large_arc = int(round(large_arc))
            sweep = int(round(sweep))
            nx, ny = cx + dx, cy + dy
            arc_p = _arc_endpoint_to_center(cx, cy, rx, ry, phi, large_arc, sweep, nx, ny)
            seg = Segment('A', [(cx, cy), (nx, ny)], arc_params=arc_p)
            current_segments.append(seg)
            cx, cy = nx, ny

        else:
            i += 1
            last_cmd = cmd
            continue

        last_cmd = cmd

        # Check if more numbers follow for implicit repeated commands
        # (handled by the loop re-using last_cmd)

    if current_segments:
        subpaths.append(current_segments)

    return subpaths


# ---------------------------------------------------------------------------
# SVG transform parsing
# ---------------------------------------------------------------------------

def _parse_transform(t_str: str) -> List[Tuple[float, ...]]:
    """Parse an SVG transform attribute into a 3x3 affine matrix [a,b,c,d,e,f]."""
    if not t_str:
        return [1, 0, 0, 1, 0, 0]  # identity

    mat = [1, 0, 0, 1, 0, 0]

    def multiply(m1, m2):
        # m = [a,b,c,d,e,f] represents matrix [[a,c,e],[b,d,f],[0,0,1]]
        a1, b1, c1, d1, e1, f1 = m1
        a2, b2, c2, d2, e2, f2 = m2
        return [
            a1*a2 + c1*b2, b1*a2 + d1*b2,
            a1*c2 + c1*d2, b1*c2 + d1*d2,
            a1*e2 + c1*f2 + e1, b1*e2 + d1*f2 + f1,
        ]

    transforms = re.findall(r'(\w+)\s*\(([^)]*)\)', t_str)
    for func, args_str in transforms:
        nums = [float(x) for x in _NUM_RE.findall(args_str)]
        if func == 'translate':
            tx = nums[0] if len(nums) > 0 else 0
            ty = nums[1] if len(nums) > 1 else 0
            mat = multiply(mat, [1, 0, 0, 1, tx, ty])
        elif func == 'scale':
            sx = nums[0] if len(nums) > 0 else 1
            sy = nums[1] if len(nums) > 1 else sx
            mat = multiply(mat, [sx, 0, 0, sy, 0, 0])
        elif func == 'rotate':
            a = math.radians(nums[0]) if len(nums) > 0 else 0
            ca, sa = math.cos(a), math.sin(a)
            if len(nums) == 3:
                cx, cy = nums[1], nums[2]
                mat = multiply(mat, [1, 0, 0, 1, cx, cy])
                mat = multiply(mat, [ca, sa, -sa, ca, 0, 0])
                mat = multiply(mat, [1, 0, 0, 1, -cx, -cy])
            else:
                mat = multiply(mat, [ca, sa, -sa, ca, 0, 0])
        elif func == 'matrix':
            if len(nums) == 6:
                mat = multiply(mat, nums)
        elif func == 'skewX':
            a = math.radians(nums[0]) if nums else 0
            mat = multiply(mat, [1, 0, math.tan(a), 1, 0, 0])
        elif func == 'skewY':
            a = math.radians(nums[0]) if nums else 0
            mat = multiply(mat, [1, math.tan(a), 0, 1, 0, 0])

    return mat


def _apply_transform_pt(mat, pt):
    a, b, c, d, e, f = mat
    x, y = pt
    return (a * x + c * y + e, b * x + d * y + f)


def _apply_transform_segments(mat, segments: List[Segment]) -> List[Segment]:
    """Apply an affine transform to all points in a list of segments."""
    identity = [1, 0, 0, 1, 0, 0]
    if mat == identity:
        return segments
    result = []
    for seg in segments:
        new_pts = [_apply_transform_pt(mat, p) for p in seg.pts]
        if seg.kind == 'A' and seg.arc_params:
            # For arcs under transform, re-sample as a cubic approximation
            # This is simpler and correct for arbitrary transforms
            result.extend(_arc_to_cubics(seg, mat))
        else:
            result.append(Segment(seg.kind, new_pts, seg.arc_params))
    return result


def _arc_to_cubics(seg: Segment, mat, n_segs=8) -> List[Segment]:
    """Approximate a transformed arc as cubic bezier segments."""
    p = seg.arc_params
    if not p or 'cx' not in p:
        new_pts = [_apply_transform_pt(mat, pt) for pt in seg.pts]
        return [Segment('L', new_pts)]

    cos_phi = math.cos(math.radians(p['phi']))
    sin_phi = math.sin(math.radians(p['phi']))

    def arc_pt(angle_deg):
        a = math.radians(angle_deg)
        x = p['cx'] + p['rx'] * math.cos(a) * cos_phi - p['ry'] * math.sin(a) * sin_phi
        y = p['cy'] + p['rx'] * math.cos(a) * sin_phi + p['ry'] * math.sin(a) * cos_phi
        return _apply_transform_pt(mat, (x, y))

    cubics = []
    prev = arc_pt(p['theta1'])
    for i in range(1, n_segs + 1):
        t = i / n_segs
        cur = arc_pt(p['theta1'] + p['dtheta'] * t)
        # Simple chord approximation as line (good enough for edge matching)
        cubics.append(Segment('L', [prev, cur]))
        prev = cur
    return cubics


# ---------------------------------------------------------------------------
# Extract closed paths from SVG elements
# ---------------------------------------------------------------------------

SVG_NS = '{http://www.w3.org/2000/svg}'


def _strip_ns(tag: str) -> str:
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def _shape_to_d(elem) -> Optional[str]:
    """Convert basic SVG shapes to path 'd' strings."""
    tag = _strip_ns(elem.tag)

    if tag == 'rect':
        x = float(elem.get('x', 0))
        y = float(elem.get('y', 0))
        w = float(elem.get('width', 0))
        h = float(elem.get('height', 0))
        if w == 0 or h == 0:
            return None
        return f'M{x},{y} L{x+w},{y} L{x+w},{y+h} L{x},{y+h} Z'

    elif tag == 'circle':
        cx = float(elem.get('cx', 0))
        cy = float(elem.get('cy', 0))
        r = float(elem.get('r', 0))
        if r == 0:
            return None
        return (f'M{cx-r},{cy} '
                f'A{r},{r} 0 1 1 {cx+r},{cy} '
                f'A{r},{r} 0 1 1 {cx-r},{cy} Z')

    elif tag == 'ellipse':
        cx = float(elem.get('cx', 0))
        cy = float(elem.get('cy', 0))
        rx = float(elem.get('rx', 0))
        ry = float(elem.get('ry', 0))
        if rx == 0 or ry == 0:
            return None
        return (f'M{cx-rx},{cy} '
                f'A{rx},{ry} 0 1 1 {cx+rx},{cy} '
                f'A{rx},{ry} 0 1 1 {cx-rx},{cy} Z')

    elif tag == 'polygon':
        points = elem.get('points', '').strip()
        if not points:
            return None
        nums = _NUM_RE.findall(points)
        if len(nums) < 4:
            return None
        d = f'M{nums[0]},{nums[1]}'
        for j in range(2, len(nums) - 1, 2):
            d += f' L{nums[j]},{nums[j+1]}'
        d += ' Z'
        return d

    elif tag == 'polyline':
        points = elem.get('points', '').strip()
        if not points:
            return None
        nums = _NUM_RE.findall(points)
        if len(nums) < 4:
            return None
        d = f'M{nums[0]},{nums[1]}'
        for j in range(2, len(nums) - 1, 2):
            d += f' L{nums[j]},{nums[j+1]}'
        return d

    elif tag == 'line':
        x1 = float(elem.get('x1', 0))
        y1 = float(elem.get('y1', 0))
        x2 = float(elem.get('x2', 0))
        y2 = float(elem.get('y2', 0))
        return f'M{x1},{y1} L{x2},{y2}'

    return None


def _collect_paths(elem, parent_transform=None) -> List[List[Segment]]:
    """Recursively collect all closed paths from SVG element tree."""
    if parent_transform is None:
        parent_transform = [1, 0, 0, 1, 0, 0]

    t_str = elem.get('transform', '')
    local_mat = _parse_transform(t_str) if t_str else [1, 0, 0, 1, 0, 0]

    def multiply(m1, m2):
        a1, b1, c1, d1, e1, f1 = m1
        a2, b2, c2, d2, e2, f2 = m2
        return [
            a1*a2 + c1*b2, b1*a2 + d1*b2,
            a1*c2 + c1*d2, b1*c2 + d1*d2,
            a1*e2 + c1*f2 + e1, b1*e2 + d1*f2 + f1,
        ]

    mat = multiply(parent_transform, local_mat)

    tag = _strip_ns(elem.tag)
    all_paths = []

    # Skip non-visible elements
    display = elem.get('display', '')
    if display == 'none':
        return []
    visibility = elem.get('visibility', '')
    if visibility == 'hidden':
        return []

    if tag == 'path':
        d = elem.get('d', '')
        if d:
            subpaths = parse_path_d(d)
            for sp in subpaths:
                if sp:
                    transformed = _apply_transform_segments(mat, sp)
                    all_paths.append(transformed)

    elif tag in ('rect', 'circle', 'ellipse', 'polygon', 'polyline', 'line'):
        d = _shape_to_d(elem)
        if d:
            subpaths = parse_path_d(d)
            for sp in subpaths:
                if sp:
                    transformed = _apply_transform_segments(mat, sp)
                    all_paths.append(transformed)

    # Recurse into children
    for child in elem:
        all_paths.extend(_collect_paths(child, mat))

    return all_paths


# ---------------------------------------------------------------------------
# SVG bounding box and scale factor
# ---------------------------------------------------------------------------

def _get_svg_dimensions(root) -> Tuple[float, float, float, float]:
    """Get the SVG viewBox or width/height as (min_x, min_y, width, height)."""
    vb = root.get('viewBox', '')
    if vb:
        parts = _NUM_RE.findall(vb)
        if len(parts) == 4:
            return tuple(float(p) for p in parts)

    w = root.get('width', '')
    h = root.get('height', '')
    if w and h:
        wn = _NUM_RE.findall(w)
        hn = _NUM_RE.findall(h)
        if wn and hn:
            return (0, 0, float(wn[0]), float(hn[0]))

    return (0, 0, 100, 100)  # fallback


def compute_scale(svg_dim: Tuple[float, float, float, float],
                  real_size: float,
                  direction: str) -> float:
    """Compute SVG units → real-world units scale factor.
    direction: 'h' for horizontal, 'v' for vertical."""
    _, _, w, h = svg_dim
    if direction.lower().startswith('h'):
        return real_size / w if w else 1.0
    else:
        return real_size / h if h else 1.0


# ---------------------------------------------------------------------------
# Edge classification: interior vs. exterior
# ---------------------------------------------------------------------------

def classify_edges(all_paths: List[List[Segment]], tol: float = 0.5):
    """Classify each segment as interior (shared by 2 paths) or exterior.

    Returns:
        interior_length: total length of interior joints (in SVG units)
        exterior_length: total length of exterior edges (in SVG units)
        total_foil_length: all piece perimeters (for foiling every piece edge)
        piece_count: number of pieces
        per_piece_perimeters: list of perimeter lengths per piece
    """
    # Build a map: canonical_key → list of (path_index, segment)
    edge_map: Dict[tuple, list] = defaultdict(list)
    per_piece_perimeters = []

    for path_idx, segments in enumerate(all_paths):
        piece_perim = 0.0
        for seg in segments:
            key = seg.canon_key(tol=tol)
            edge_map[key].append((path_idx, seg))
            piece_perim += seg.length()
        per_piece_perimeters.append(piece_perim)

    interior_length = 0.0
    exterior_length = 0.0

    for key, entries in edge_map.items():
        seg_length = entries[0][1].length()
        unique_paths = len(set(e[0] for e in entries))
        if unique_paths >= 2:
            # Interior edge: shared between 2+ pieces
            interior_length += seg_length
        else:
            # Exterior edge: belongs to only one piece
            exterior_length += seg_length

    return interior_length, exterior_length, sum(per_piece_perimeters), len(all_paths), per_piece_perimeters


# ---------------------------------------------------------------------------
# Cost calculations
# ---------------------------------------------------------------------------

def calculate_costs(interior_length_units: float,
                    exterior_length_units: float,
                    total_perimeter_units: float,
                    unit_label: str,
                    foil_width_inches: float,
                    foil_backing: str = "copper",
                    units_per_inch: float = 1.0):
    """Calculate foil and solder costs.

    Foil is applied to every piece edge (full perimeter of each piece).
    For exterior edges, only ~3mm wraps onto each face, but the foil still
    runs the full length of the edge – it's just trimmed/hidden under the frame.
    So total foil consumed ≈ total perimeter of all pieces.

    Solder is applied only to interior joints, on both front and back.

    Args:
        interior_length_units: interior joint length in user units
        exterior_length_units: exterior edge length in user units
        total_perimeter_units: sum of all piece perimeters in user units
        unit_label: 'inches' or 'cm'
        foil_width_inches: chosen foil width
        foil_backing: 'copper', 'black', or 'silver'
        units_per_inch: conversion factor (1.0 for inches, 2.54 for cm)
    """
    # Convert everything to inches for cost lookup
    interior_inches = interior_length_units / units_per_inch
    total_perim_inches = total_perimeter_units / units_per_inch

    # Apply 5% waste factor
    foil_needed_inches = total_perim_inches * WASTE_FACTOR
    solder_joint_inches = interior_inches * WASTE_FACTOR

    # Find the best matching foil in catalog
    foil_key = None
    # Try exact width match first
    for (w, backing), info in FOIL_CATALOG.items():
        if abs(w - foil_width_inches) < 0.01 and backing == foil_backing:
            foil_key = (w, backing)
            break

    # Fallback: try any backing with matching width
    if foil_key is None:
        for (w, backing), info in FOIL_CATALOG.items():
            if abs(w - foil_width_inches) < 0.01:
                foil_key = (w, backing)
                break

    # Fallback: find closest width
    if foil_key is None:
        closest = min(FOIL_CATALOG.keys(),
                      key=lambda k: abs(k[0] - foil_width_inches))
        foil_key = closest

    foil_price, foil_brand, roll_length = FOIL_CATALOG[foil_key]

    rolls_needed = math.ceil(foil_needed_inches / roll_length)
    foil_cost = rolls_needed * foil_price

    # Solder: interior joints × 2 sides (front and back)
    solder_linear_feet = (solder_joint_inches * 2) / 12.0  # two sides, convert to feet
    solder_lbs = solder_linear_feet * SOLDER_LBS_PER_FOOT_OF_SEAM
    solder_cost = solder_lbs * SOLDER_PRICE_PER_LB

    return {
        'foil_needed_inches': foil_needed_inches,
        'foil_needed_units': foil_needed_inches * units_per_inch,
        'foil_key': foil_key,
        'foil_brand': foil_brand,
        'foil_price_per_roll': foil_price,
        'foil_roll_length_inches': roll_length,
        'rolls_needed': rolls_needed,
        'foil_cost': foil_cost,
        'solder_joint_inches': solder_joint_inches,
        'solder_linear_feet_both_sides': solder_linear_feet,
        'solder_lbs': solder_lbs,
        'solder_cost': solder_cost,
        'total_material_cost': foil_cost + solder_cost,
    }


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

def select_svg_file(path_arg: Optional[str] = None) -> str:
    """Select an SVG file via argument or file dialog."""
    if path_arg and os.path.isfile(path_arg):
        return path_arg

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        # Start on Desktop if it exists
        desktop = os.path.expanduser("~/Desktop")
        initial_dir = desktop if os.path.isdir(desktop) else os.path.expanduser("~")
        filepath = filedialog.askopenfilename(
            title="Select Stained Glass SVG Pattern",
            initialdir=initial_dir,
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
        )
        root.destroy()
        if not filepath:
            print("No file selected. Exiting.")
            sys.exit(0)
        return filepath
    except ImportError:
        print("tkinter not available. Please provide the SVG path as a command-line argument:")
        print(f"  python {sys.argv[0]} <path_to_svg>")
        sys.exit(1)


# ---------------------------------------------------------------------------
# User input helpers
# ---------------------------------------------------------------------------

def get_dimension_input():
    """Prompt user for the real-world dimension of the design."""
    print("\n--- Real-World Dimension ---")
    print("Provide one known dimension of the finished piece.")

    while True:
        direction = input("Dimension direction — (h)orizontal or (v)ertical? [h]: ").strip().lower()
        if direction in ('', 'h', 'horizontal'):
            direction = 'h'
            break
        elif direction in ('v', 'vertical'):
            direction = 'v'
            break
        print("Please enter 'h' or 'v'.")

    while True:
        try:
            size = float(input(f"{'Horizontal' if direction == 'h' else 'Vertical'} measurement: "))
            if size > 0:
                break
            print("Must be positive.")
        except ValueError:
            print("Enter a number.")

    while True:
        unit = input("Unit — (in)ches or (cm)? [in]: ").strip().lower()
        if unit in ('', 'in', 'inches', 'inch'):
            return size, 'inches', direction, 1.0
        elif unit in ('cm', 'centimeters', 'centimeter'):
            return size, 'cm', direction, 1.0 / 2.54  # units_per_inch = 1/2.54 means 1 inch = 2.54 cm
        print("Please enter 'in' or 'cm'.")


def get_foil_input():
    """Prompt user for foil width and backing preference."""
    print("\n--- Foil Selection ---")
    print("Common widths: 3/16\" (0.1875), 7/32\" (0.21875), 1/4\" (0.25)")
    print("Common backings: copper, black, silver")

    while True:
        w_str = input("Foil width in inches [7/32]: ").strip()
        if not w_str:
            width = 7/32
            break
        try:
            if '/' in w_str:
                num, den = w_str.split('/')
                width = float(num) / float(den)
            else:
                width = float(w_str)
            if width > 0:
                break
            print("Must be positive.")
        except (ValueError, ZeroDivisionError):
            print("Enter a decimal or fraction like 7/32.")

    while True:
        backing = input("Foil backing (copper/black/silver) [copper]: ").strip().lower()
        if not backing:
            backing = 'copper'
        if backing in ('copper', 'black', 'silver'):
            break
        print("Please enter copper, black, or silver.")

    return width, backing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Stained Glass Foil & Solder Cost Calculator")
    print("=" * 60)

    # 1. Select SVG
    svg_path = select_svg_file(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"\nLoaded: {svg_path}")

    # 2. Parse SVG
    tree = ET.parse(svg_path)
    root = tree.getroot()
    svg_dim = _get_svg_dimensions(root)
    print(f"SVG dimensions: viewBox/size = {svg_dim[2]:.1f} × {svg_dim[3]:.1f}")

    all_paths = _collect_paths(root)
    print(f"Found {len(all_paths)} closed path(s) / piece(s)")

    if len(all_paths) == 0:
        print("\nNo closed paths found in this SVG. Make sure each glass piece")
        print("is drawn as a closed path (ending with Z).")
        sys.exit(1)

    # 3. Get user inputs
    real_size, unit_label, direction, inv_units_per_inch = get_dimension_input()
    # inv_units_per_inch: for inches it's 1.0, for cm it's 1/2.54
    # We need units_per_inch: how many user-units per inch
    if unit_label == 'cm':
        units_per_inch = 2.54
    else:
        units_per_inch = 1.0

    foil_width, foil_backing = get_foil_input()

    # 4. Compute scale factor
    scale = compute_scale(svg_dim, real_size, direction)
    print(f"\nScale factor: 1 SVG unit = {scale:.6f} {unit_label}")

    # 5. Classify edges
    # Use a tolerance proportional to the SVG size (0.1% of max dimension)
    tol = max(svg_dim[2], svg_dim[3]) * 0.001
    interior_svg, exterior_svg, total_perim_svg, piece_count, piece_perims = classify_edges(all_paths, tol=tol)

    # Convert to user units
    interior = interior_svg * scale
    exterior = exterior_svg * scale
    total_perim = total_perim_svg * scale

    # 6. Calculate costs
    costs = calculate_costs(
        interior_length_units=interior,
        exterior_length_units=exterior,
        total_perimeter_units=total_perim,
        unit_label=unit_label,
        foil_width_inches=foil_width,
        foil_backing=foil_backing,
        units_per_inch=units_per_inch,
    )

    # 7. Print results
    abbr = "in" if unit_label == "inches" else "cm"
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print(f"\n  Pieces:                  {piece_count}")
    print(f"  Total perimeter (all):   {total_perim:,.1f} {abbr}")
    print(f"  Interior joints:         {interior:,.1f} {abbr}")
    print(f"  Exterior edges:          {exterior:,.1f} {abbr}")

    # Quick sanity check
    implied_interior = (total_perim - exterior) / 2
    print(f"  (Cross-check: interior via sum formula = {implied_interior:,.1f} {abbr})")

    print(f"\n--- Foil ---")
    wfrac = f"{foil_width:.4f}\""
    # Try to show as a nice fraction
    fracs = {3/16: '3/16"', 7/32: '7/32"', 1/4: '1/4"', 1/2: '1/2"',
             3/8: '3/8"', 5/16: '5/16"'}
    wfrac = fracs.get(foil_width, f'{foil_width:.4f}"')
    print(f"  Selected foil:           {costs['foil_brand']} {wfrac} {foil_backing}-backed")
    print(f"  Foil per roll:           {costs['foil_roll_length_inches']} inches ({costs['foil_roll_length_inches']/36:.0f} yards)")
    print(f"  Total foil needed:       {costs['foil_needed_units']:,.1f} {abbr}  (incl. 5% waste)")
    print(f"  Rolls needed:            {costs['rolls_needed']}")
    print(f"  Cost per roll:           ${costs['foil_price_per_roll']:.2f}")
    print(f"  FOIL COST:               ${costs['foil_cost']:.2f}")

    print(f"\n--- Solder (60/40, .125\" dia.) ---")
    print(f"  Interior joints:         {interior:,.1f} {abbr}")
    print(f"  Soldered length (×2):    {costs['solder_linear_feet_both_sides']:,.1f} ft  (front + back, incl. 5% waste)")
    print(f"  Estimated solder:        {costs['solder_lbs']:.2f} lbs")
    print(f"  Price per lb:            ${SOLDER_PRICE_PER_LB:.2f}  (Amerway 60/40 5-pack = $117.50)")
    print(f"  SOLDER COST:             ${costs['solder_cost']:.2f}")

    print(f"\n{'=' * 60}")
    print(f"  TOTAL MATERIALS:         ${costs['total_material_cost']:.2f}")
    print(f"{'=' * 60}")

    print(f"\nNotes:")
    print(f"  • Foil estimate covers the full perimeter of every piece (each")
    print(f"    edge gets foiled whether interior or exterior).")
    print(f"  • Exterior edges will be trimmed/hidden under the frame – only")
    print(f"    ~3mm wraps onto each face – but the foil tape still runs the")
    print(f"    full length, so the material usage is the same.")
    print(f"  • Solder estimate uses the rule of thumb: ~1 lb per 16 linear")
    print(f"    feet of finished bead. Your actual usage will vary with bead")
    print(f"    width and technique.")
    print(f"  • Prices fetched from stainedglassforless.com (foil) and")
    print(f"    pacwestsales.com (solder) as of Feb 2026. Verify before buying.")


if __name__ == '__main__':
    main()