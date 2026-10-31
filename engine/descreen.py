#!/usr/bin/env python3
"""Descreen offset-printed CD fronts for display and FLAC embed.

Takes a Pixelmator-processed JPEG (already straightened, cropped, WhiBal'd)
and notches the CMYK halftone screen. Archival MusicBrainz JPEGs stay as-is.
Writes the full-res result plus a max-600px (longer side) copy for players.

Mirrors Sattva's two handwritten passes: (1) remove the screen lattice and
its harmonics, (2) reduce the colour rosette — then apply that only where
the image actually looks screened, so type stays out of the filter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    maximum_filter,
    median_filter,
)


def load_rgb(path: Path) -> tuple[np.ndarray, dict]:
    im = Image.open(path)
    meta = {
        "icc_profile": im.info.get("icc_profile"),
        "dpi": im.info.get("dpi", (600.0, 600.0)),
        "mode": im.mode,
        "size": im.size,
    }
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if im.mode == "L":
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = np.repeat(arr[..., None], 3, axis=2)
    else:
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr, meta


def save_jpeg(
    path: Path,
    rgb: np.ndarray,
    *,
    quality: int,
    icc_profile: bytes | None,
    dpi: tuple,
) -> None:
    rgb = np.clip(rgb, 0.0, 1.0)
    im = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    kwargs = {
        "quality": quality,
        "subsampling": 0,
        "optimize": True,
        "dpi": (float(dpi[0]), float(dpi[1])),
    }
    if icc_profile:
        kwargs["icc_profile"] = icc_profile
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", **kwargs)


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (-0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = 0.5 + (0.5 * r - 0.418688 * g - 0.081312 * b)
    return y.astype(np.float32), cb.astype(np.float32), cr.astype(np.float32)


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    return np.clip(np.stack((r, g, b), axis=-1), 0.0, 1.0).astype(np.float32)


def _freq_lpi(h: int, w: int, dpi: float) -> np.ndarray:
    cy, cx = h / 2.0, w / 2.0
    y = (np.arange(h, dtype=np.float32)[:, None] - cy) / h
    x = (np.arange(w, dtype=np.float32)[None, :] - cx) / w
    return np.hypot(y, x) * dpi


def detect_screen(gray: np.ndarray, dpi: float, lpi_range: tuple[float, float]) -> dict:
    """Measure LPI from midtones so solid type doesn't dominate the spectrum."""
    h, w = gray.shape
    mid = np.clip(1.0 - np.abs(gray - 0.42) / 0.40, 0.0, 1.0)
    mid = gaussian_filter(mid, 8.0)
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32) * mid
    mag = np.abs(fftshift(fft2(gray * win, workers=-1))).astype(np.float32)
    r_lpi = _freq_lpi(h, w, dpi)
    mag_n = mag * np.sqrt(r_lpi / max(dpi, 1.0) * max(h, w) + 1.0)
    mag_s = gaussian_filter(mag_n, 1.0)

    lo, hi = lpi_range
    band = (r_lpi >= lo) & (r_lpi <= hi)
    peaks = (mag_s == maximum_filter(mag_s, size=11)) & band
    ys, xs = np.where(peaks)
    cy, cx = h / 2.0, w / 2.0
    fallback = {
        "lpi": 150.0,
        "period_px": dpi / 150.0,
        "angles": [0.0, 15.0, 45.0, 75.0],
        "freqs": [],
        "n_peaks": 0,
        "ok": False,
    }
    if ys.size == 0:
        return fallback

    order = np.argsort(mag_s[ys, xs])[::-1]
    keep = order[: min(32, order.size)]
    ys, xs = ys[keep], xs[keep]
    peak_lpi = r_lpi[ys, xs]
    strongest = peak_lpi[: max(4, len(peak_lpi) // 2)]
    lpi = float(np.clip(np.median(strongest), lo, hi))

    tight = (r_lpi > lpi * 0.82) & (r_lpi < lpi * 1.18)
    peak_ann = (mag_s == maximum_filter(mag_s, size=9)) & tight
    p_y, p_x = np.where(peak_ann)
    freqs: list[tuple[float, float]] = []
    if p_y.size:
        p_order = np.argsort(mag_s[p_y, p_x])[::-1][:20]
        freqs = [((p_x[i] - cx) / w, (p_y[i] - cy) / h) for i in p_order]

    measured: list[float] = []
    for fx, fy in freqs:
        ang = (np.degrees(np.arctan2(fy, fx)) + 180.0) % 90.0  # lattice is 90°-periodic
        if all(min(abs(ang - a), 90.0 - abs(ang - a)) > 7.0 for a in measured):
            measured.append(ang)
        if len(measured) == 4:
            break
    standard = [0.0, 15.0, 45.0, 75.0]
    angles = list(standard)
    for a in measured:
        snapped = min(standard, key=lambda s: min(abs(a - s), 90.0 - abs(a - s)))
        if min(abs(a - snapped), 90.0 - abs(a - snapped)) > 8.0:
            angles.append(float(a))

    return {
        "lpi": lpi,
        "period_px": dpi / lpi,
        "angles": angles[:6],
        "freqs": freqs,
        "n_peaks": int(len(freqs)),
        "ok": True,
    }


def _add_notch(holes: np.ndarray, px: float, py: float, sig_x: float, sig_y: float) -> None:
    h, w = holes.shape
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    g = np.exp(-0.5 * (((x - px) / sig_x) ** 2 + ((y - py) / sig_y) ** 2))
    np.maximum(holes, g, out=holes)


def geometric_notch_mask(
    h: int,
    w: int,
    *,
    dpi: float,
    lpi: float,
    angles: list[float],
    freqs: list[tuple[float, float]],
    atten: float,
    notch_lpi: float,
    lattice: int = 2,
    beats: bool = False,
    beat_min_freq: float = 0.0,
) -> np.ndarray:
    """Soft mask (1 = keep). Lattice peaks (n,m)·S, never a hard zero."""
    cy, cx = h / 2.0, w / 2.0
    holes = np.zeros((h, w), dtype=np.float32)
    sig_x = max((notch_lpi / dpi) * w, 3.0)
    sig_y = max((notch_lpi / dpi) * h, 3.0)
    nyquist = 0.48

    def punch(fx: float, fy: float, min_f: float = 0.0) -> None:
        f = float(np.hypot(fx, fy))
        if f < max(1e-6, min_f) or f >= nyquist:
            return
        _add_notch(holes, cx + fx * w, cy + fy * h, sig_x, sig_y)
        _add_notch(holes, cx - fx * w, cy - fy * h, sig_x, sig_y)

    s = lpi / dpi
    for ang in angles:
        th = np.deg2rad(ang)
        v1x, v1y = s * np.cos(th), s * np.sin(th)
        v2x, v2y = -s * np.sin(th), s * np.cos(th)
        for n in range(-lattice, lattice + 1):
            for m in range(-lattice, lattice + 1):
                if n == 0 and m == 0:
                    continue
                if n < 0 or (n == 0 and m < 0):
                    continue
                punch(n * v1x + m * v2x, n * v1y + m * v2y)

    for fx, fy in freqs:
        punch(fx, fy)
        punch(2.0 * fx, 2.0 * fy)

    if beats and len(angles) >= 2:
        # Pairwise CMYK interference (rosette). Skip the ~18px 0°–15° beat
        # (face lighting / grain clumps); keep the ~9px hex.
        vecs = [(s * np.cos(np.deg2rad(a)), s * np.sin(np.deg2rad(a))) for a in angles]
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                punch(vecs[i][0] - vecs[j][0], vecs[i][1] - vecs[j][1], beat_min_freq)
                punch(vecs[i][0] + vecs[j][0], vecs[i][1] + vecs[j][1], beat_min_freq)

    return 1.0 - np.clip(holes, 0.0, 1.0) * atten


def fft_notch_channels(channels: list[np.ndarray], mask: np.ndarray, pad: int = 64) -> list[np.ndarray]:
    out = []
    for ch in channels:
        padded = np.pad(ch, pad, mode="reflect")
        if mask.shape != padded.shape:
            ph, pw = padded.shape
            # Caller should pass a mask matching the padded size.
            raise ValueError(f"mask {mask.shape} != padded channel {padded.shape}")
        F = fftshift(fft2(padded, workers=-1))
        F *= mask
        rec = ifft2(ifftshift(F), workers=-1).real.astype(np.float32)
        out.append(rec[pad:-pad, pad:-pad] if pad else rec)
    return out


def screen_amount(rgb: np.ndarray, period_px: float) -> np.ndarray:
    """1 where the original looks screened, 0 on solid type / paper / logos.

    Photos and drop shadows are mid-grey with oscillation at the screen period.
    Large-scale shading (a face, a fold) must not suppress that — otherwise the
    gate blends the raw dots back in, which is how v6 under-descreened.
    """
    y = luma(rgb)
    dog = np.abs(
        gaussian_filter(y, max(period_px * 0.22, 0.7))
        - gaussian_filter(y, max(period_px * 0.95, 1.8))
    )
    # Solid black/white ink and paper are not screened; keep them out.
    mid = np.clip((y - 0.04) / 0.05, 0.0, 1.0) * np.clip((0.97 - y) / 0.05, 0.0, 1.0)
    raw = gaussian_filter(dog * mid, max(period_px * 0.45, 1.2))
    lo, hi = np.percentile(raw, (8.0, 52.0))
    m = np.clip((raw - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return np.clip(m * 1.2, 0.0, 1.0)


def color_edge(y: np.ndarray, cb: np.ndarray, cr: np.ndarray, period_px: float) -> np.ndarray:
    """1 at intended ink/colour edges, 0 on screened flats.

    Gradients are taken after a sub-period blur so CMYK dots do not light up;
    comic swooshes and type still do.
    """
    s = max(period_px * 0.3, 0.7)
    gc = np.hypot(
        np.hypot(*np.gradient(gaussian_filter(cb, s))),
        np.hypot(*np.gradient(gaussian_filter(cr, s))),
    ).astype(np.float32)
    gy = np.hypot(*np.gradient(gaussian_filter(y, s))).astype(np.float32)
    raw = np.maximum(gc * 3.0, gy * 0.45)
    lo = float(np.percentile(raw, 72.0))
    hi = float(np.percentile(raw, 96.5))
    e = np.clip((raw - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    e = gaussian_filter(e, max(period_px * 0.22, 0.5))
    return np.clip(e * 1.2, 0.0, 1.0).astype(np.float32)


def _norm_blur(ch: np.ndarray, weight: np.ndarray, sig: float) -> np.ndarray:
    """Gaussian blur that will not pull values across weight=0 barriers."""
    den = gaussian_filter(weight, sig)
    return gaussian_filter(ch * weight, sig) / np.maximum(den, 1e-4)


def chroma_rosette(
    cb: np.ndarray,
    cr: np.ndarray,
    period_px: float,
    amount: float,
    gate: np.ndarray,
    edge: np.ndarray,
    sig_scale: float = 1.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Sattva pass 2 analogue: smooth colour blotch, leave luma grain.

    Neighbourhood blur only runs where the pixel's hue already matches the
    neighbourhood, so cream type on a pink sky is not pulled toward the sky.
    Near-grey pixels always mix (no hue to protect). Colour edges (blue
    swoosh on red) are a barrier, so ink does not bleed across them.

    sig_scale is in screen periods. ~1.25 only kills local rosette cells;
    ~1.8 also flattens the longer beat/corduroy that FFT lattice notches
    leave in large screened fills (e.g. yellow Yes-logo fields).
    """
    if amount <= 0:
        return cb, cr
    sig = max(period_px * sig_scale, 1.6)
    hold = np.clip(1.0 - edge, 0.0, 1.0)
    src_w = (hold * hold).astype(np.float32)
    cb_b = _norm_blur(cb, src_w, sig)
    cr_b = _norm_blur(cr, src_w, sig)
    vox, voy = cb - 0.5, cr - 0.5
    vbx, vby = cb_b - 0.5, cr_b - 0.5
    mag = np.hypot(vox, voy)
    align = (vox * vbx + voy * vby) / (mag * np.hypot(vbx, vby) + 1e-6)
    # 1 on grey or same-hue; 0 when the blur would rotate hue.
    hue_w = np.clip(align, 0.0, 1.0) * (mag / (mag + 0.02)) + (0.02 / (mag + 0.02))
    w = (gate * amount * hold * hue_w).astype(np.float32)
    cb = cb + w * (cb_b - cb)
    cr = cr + w * (cr_b - cr)
    return cb, cr


def restore_chroma_amount(
    cb_p: np.ndarray,
    cr_p: np.ndarray,
    cb_o: np.ndarray,
    cr_o: np.ndarray,
    period_px: float,
    edge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep processed hue; match local |chroma| to the original.

    Flattening the rosette averages ink dots with paper, so a comic red
    field drops below the saturation you see on the scan. Scaling back to
    the original's local chroma magnitude restores that punch without
    bringing the dots back. Magnitude is estimated inside each colour
    region so a blue stripe does not donate saturation to neighbouring red.
    """
    sig = max(period_px * 1.55, 2.2)
    hold = np.clip(1.0 - edge, 0.0, 1.0).astype(np.float32)
    src_w = hold * hold
    mag_p = np.hypot(cb_p - 0.5, cr_p - 0.5)
    mag_o = np.hypot(cb_o - 0.5, cr_o - 0.5)
    num = _norm_blur(mag_o, src_w, sig)
    den = _norm_blur(mag_p, src_w, sig)
    scale = np.clip(num / np.maximum(den, 1e-4), 0.85, 1.35).astype(np.float32)
    scale = 1.0 + (scale - 1.0) * hold
    return 0.5 + (cb_p - 0.5) * scale, 0.5 + (cr_p - 0.5) * scale


def luma_residual(
    y: np.ndarray,
    period_px: float,
    amount: float,
    gate: np.ndarray,
    edge: np.ndarray,
) -> np.ndarray:
    """Flatten leftover luma dots at one screen period. Edges are a barrier."""
    if amount <= 0:
        return y
    sig = max(period_px * 1.0, 1.2)
    hold = np.clip(1.0 - edge, 0.0, 1.0)
    w = (gate * amount * hold).astype(np.float32)
    return y + w * (gaussian_filter(y, sig) - y)


def luma_median(
    y: np.ndarray,
    amount: float,
    gate: np.ndarray,
    edge: np.ndarray,
    size: int = 5,
) -> np.ndarray:
    """Break the hex rosette without a big gaussian. Median kills the regular
    cell and leaves grain; edges are a barrier so type stays sharp.
    """
    if amount <= 0:
        return y
    hold = np.clip(1.0 - edge, 0.0, 1.0)
    w = (gate * amount * hold).astype(np.float32)
    return y + w * (median_filter(y, size=size) - y)


def mild_sharpen(
    rgb: np.ndarray,
    amount: float,
    radius: float = 0.65,
    edge: np.ndarray | None = None,
) -> np.ndarray:
    """Unsharp mask. If edge is given, sharpen only away from hard ink so we
    don't amplify FFT ghost outlines parallel to logo strokes / type."""
    if amount <= 0:
        return rgb
    blur = gaussian_filter(rgb, sigma=(radius, radius, 0.0))
    sharp = rgb + amount * (rgb - blur)
    if edge is not None:
        # 1 in flats, ~0 on ink edges (and a short skirt).
        w = np.clip(1.0 - gaussian_filter(edge, 2.0) * 1.5, 0.0, 1.0).astype(np.float32)
        sharp = sharp * w[..., None] + rgb * (1.0 - w[..., None])
    return np.clip(sharp, 0.0, 1.0)


def edge_antiring(
    ch: np.ndarray,
    eband: np.ndarray,
    amount: float,
    sig: float = 1.2,
) -> np.ndarray:
    """Legacy: pull FFT toward a local mean inside a soft edge band.

    Softens ink edges — kept for CLI experiments. Prefer suppress_edge_ring.
    """
    if amount <= 0:
        return ch
    return ch + eband * amount * (gaussian_filter(ch, sig) - ch)


def _print_ink_mask(
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    edge: np.ndarray,
    rgb: np.ndarray | None,
    *,
    hard_thr: float = 0.32,
) -> np.ndarray:
    """Black outlines + deep red strokes. Skips orange text fills (mid green,
    Y≳0.45) and green/blue photo edges so skirts don't mottle lettering."""
    y_s = gaussian_filter(y, 2.0)
    hard = edge > hard_thr
    near_hard = binary_dilation(hard, iterations=1)
    black = near_hard & (y < 0.42) & (y < y_s - 0.035)
    if rgb is None:
        return black
    chroma = np.sqrt(cb * cb + cr * cr)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = (
        near_hard
        & (cr > 0.10)
        & (chroma > 0.12)
        & (y < 0.40)
        & (y < y_s - 0.02)
        & (r > g + 0.15)
        & (r > b + 0.15)
        & (g < 0.28)
    )
    return black | red


def suppress_edge_ring(
    y_fft: np.ndarray,
    cb_fft: np.ndarray,
    cr_fft: np.ndarray,
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    edge: np.ndarray,
    period_px: float,
    *,
    rgb: np.ndarray | None = None,
    amount: float = 1.0,
    outer_periods: float = 4.0,
    field_erode: int = 5,
    field_sig: float = 3.5,
    band_boost: float = 1.2,
    outer_w: float = 0.15,
    y_morph: float = 0.12,
    y_kill: float = 0.3,
    chroma_to_orig: float = 0.5,
    core: float = 0.85,
    spat_sig_y: float = 1.1,
    hf_sig: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kill FFT Gibbs ghosts just outside hard ink, keep stroke sharpness.

    Geometric notches ring parallel to ink — e.g. a dark band in the yellow
    fill 2–3px outside the Yes "S" red stroke. Edge-band blur smears the
    stroke; chroma-morph toward a spatial blur invents cyan fringes. Instead:

    1. Detect print ink = black outlines OR red strokes (RGB-gated so green
       landscape edges don't qualify).
    2. Estimate a flat field from the FFT result with ink eroded out.
    3. Full-strength clamp-to-field in the 1.5–3.5px ghost band; light touch
       on a wider skirt. Never cross the field (avoids white lift blotches).
    4. Restore original chroma in the weight mask; put sharp ink back on core.
    """
    if amount <= 0:
        return y_fft, cb_fft, cr_fft

    dark = _print_ink_mask(y, cb, cr, edge, rgb)
    if int(dark.sum()) < 16:
        return y_fft, cb_fft, cr_fft

    outer = max(int(round(period_px * outer_periods)), 8)
    dist = distance_transform_edt(~dark)
    near_ink = np.clip(gaussian_filter(dark.astype(np.float32), 3.0) * 2.2, 0.0, 1.0)
    skirt = (dist > 0) & (dist <= outer)
    sk = np.clip(gaussian_filter(skirt.astype(np.float32), 1.1), 0.0, 1.0) * near_ink
    # Ghost lives ~2px out; stay off the 0–1.2px AA fringe (lifting there
    # sparks pale flecks right on the stroke).
    band = (
        gaussian_filter(((dist >= 1.2) & (dist <= 3.5)).astype(np.float32), 0.55)
        * near_ink
    )
    w = np.clip(band_boost * band + outer_w * sk, 0.0, 1.0).astype(np.float32) * amount
    if float(w.max()) < 1e-4:
        return y_fft, cb_fft, cr_fft

    dark_ex = binary_dilation(dark, iterations=max(field_erode, 0))
    y_field = _norm_blur(y_fft, (~dark_ex).astype(np.float32), field_sig)

    hold = np.clip(1.0 - edge, 0.0, 1.0).astype(np.float32)
    hold = hold * hold
    y_sp = _norm_blur(median_filter(y, 3), hold, max(period_px * spat_sig_y, 1.2))

    y2 = y_fft * (1.0 - y_morph * w) + y_sp * (y_morph * w)
    dy = y2 - y_sp
    y2 = y2 - y_kill * w * (dy - gaussian_filter(dy, hf_sig))
    # Clamp to field — never overshoot (lift>1 used to punch white flecks).
    up = np.clip(y_field - y2, 0.0, None)
    dn = np.clip(y2 - y_field, 0.0, None)
    y2 = y2 + w * up - w * dn

    c_m = chroma_to_orig
    cb2 = cb_fft * (1.0 - c_m * w) + cb * (c_m * w)
    cr2 = cr_fft * (1.0 - c_m * w) + cr * (c_m * w)

    if core > 0:
        y2 = y2 + core * dark.astype(np.float32) * (median_filter(y, 3) - y2)

    return y2, cb2, cr2


def kill_ring_specks(
    rgb_out: np.ndarray,
    rgb_src: np.ndarray,
    edge: np.ndarray,
    *,
    amount: float = 1.0,
    thr: float = 0.015,
    size: int = 5,
) -> np.ndarray:
    """Median-kill residual bright flecks near print ink after sharpen."""
    if amount <= 0:
        return rgb_out
    y, cb, cr = rgb_to_ycbcr(rgb_out)
    ys, cbs, crs = rgb_to_ycbcr(rgb_src)
    dark = _print_ink_mask(ys, cbs, crs, edge, rgb_src)
    if int(dark.sum()) < 16:
        return rgb_out
    dist = distance_transform_edt(~dark)
    near_ink = np.clip(gaussian_filter(dark.astype(np.float32), 3.0) * 2.0, 0.0, 1.0)
    med = median_filter(y, size)
    bright = (
        (y > med + thr)
        & (dist > 1.0)
        & (dist < 12.0)
        & (near_ink > 0.04)
        & ~dark
    )
    if not np.any(bright):
        return rgb_out
    y2 = np.where(bright, med, y)
    return np.clip(ycbcr_to_rgb(y2, cb, cr), 0.0, 1.0)


def match_ycbcr(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Undo global Y/Cb/Cr drift from clipping. Offsets keep local hue."""
    ys, cbs, crs = rgb_to_ycbcr(src)
    yr, cbr, crr = rgb_to_ycbcr(ref)
    ys = ys + (float(yr.mean()) - float(ys.mean()))
    cbs = cbs + (float(cbr.mean()) - float(cbs.mean()))
    crs = crs + (float(crr.mean()) - float(crs.mean()))
    return ycbcr_to_rgb(ys, cbs, crs)


def descreen_rgb(
    rgb: np.ndarray,
    *,
    dpi: float,
    atten: float,
    notch_lpi: float,
    tile: int,
    smooth: float,
    sharpen: float,
    strength: float,
    chroma: float,
    median: float = 0.70,
    beat_min_freq: float = 0.08,
    chroma_sig: float = 1.8,
    ring: float = 0.0,
    edge_ring_y: float = 0.0,
    edge_ring_c: float = 0.0,
    edge_ring_blur: float = 3.0,
) -> tuple[np.ndarray, dict]:
    info = detect_screen(luma(rgb), dpi=dpi, lpi_range=(90.0, 210.0))
    period = info["period_px"]
    pad = 64
    ph, pw = rgb.shape[0] + 2 * pad, rgb.shape[1] + 2 * pad

    mask_y = geometric_notch_mask(
        ph, pw, dpi=dpi, lpi=info["lpi"], angles=info["angles"],
        freqs=info["freqs"], atten=atten, notch_lpi=notch_lpi, lattice=2,
        beats=True, beat_min_freq=beat_min_freq,
    )
    mask_c = geometric_notch_mask(
        ph, pw, dpi=dpi, lpi=info["lpi"], angles=info["angles"],
        freqs=info["freqs"], atten=atten, notch_lpi=notch_lpi, lattice=2, beats=False,
    )

    y, cb, cr = rgb_to_ycbcr(rgb)
    y2, = fft_notch_channels([y], mask_y, pad=pad)
    # Chroma skips beat notches — those punch near DC and rotate cream type.
    cb2, cr2 = fft_notch_channels([cb, cr], mask_c, pad=pad)

    # Spectral screen removal is global (type isn't periodic, so it survives).
    # The extra moiré/residual flatten is gated so it only hits screened areas.
    gate = screen_amount(rgb, period)
    edge = color_edge(y, cb, cr, period)
    # Outside-ink skirt: clamp-to-field Gibbs kill (keeps stroke sharp).
    y2, cb2, cr2 = suppress_edge_ring(
        y2, cb2, cr2, y, cb, cr, edge, period, rgb=rgb, amount=ring
    )
    # Legacy soft edge-band blur (off by default — smears ink).
    if edge_ring_y > 0 or edge_ring_c > 0:
        eband = np.clip(gaussian_filter(edge, edge_ring_blur), 0.0, 1.0).astype(np.float32)
        y2 = edge_antiring(y2, eband, edge_ring_y)
        cb2 = edge_antiring(cb2, eband, edge_ring_c)
        cr2 = edge_antiring(cr2, eband, edge_ring_c)
    # FFT chroma rings at hard ink edges; put the original colour back there.
    cb2 = cb2 + edge * (cb - cb2)
    cr2 = cr2 + edge * (cr - cr2)
    barrier = np.clip(gaussian_filter(edge, max(period * 0.45, 1.0)) * 1.35, 0.0, 1.0)
    cb2, cr2 = chroma_rosette(
        cb2, cr2, period, amount=chroma, gate=gate, edge=barrier, sig_scale=chroma_sig
    )
    y2 = luma_residual(y2, period, amount=smooth, gate=gate, edge=edge)
    y2 = luma_median(y2, amount=median, gate=gate, edge=edge, size=5)

    screened = ycbcr_to_rgb(y2, cb2, cr2)
    screened = mild_sharpen(screened, amount=sharpen, edge=edge)
    screened = match_ycbcr(np.clip(screened, 0.0, 1.0), rgb)
    # Restore after the mean match — a red-heavy cover would otherwise
    # have that match pull Cr back down and undo the saturation.
    ys, cbs, crs = rgb_to_ycbcr(np.clip(screened, 0.0, 1.0))
    cbs, crs = restore_chroma_amount(cbs, crs, cb, cr, period, edge=barrier)
    screened = np.clip(ycbcr_to_rgb(ys, cbs, crs), 0.0, 1.0)
    if strength < 1.0:
        screened = rgb * (1.0 - strength) + screened * strength
    info["screen_mean"] = float(gate.mean())
    _ = tile  # reserved; whole-image FFT is the right default at CD-front size
    return screened, info


EMBED_MAX = 600  # longer side; embed fits inside EMBED_MAX×EMBED_MAX, never upscales


def resize_max(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """Fit inside max_side×max_side preserving aspect ratio; never upscale."""
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        return rgb
    scale = max_side / max(h, w)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8), "RGB")
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=0.7, percent=30, threshold=2))
    return np.asarray(im, dtype=np.float32) / 255.0


def default_output(src: Path) -> Path:
    return src.with_name(src.stem + "-descreen.jpg")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Descreen a Pixelmator-processed CD front. Writes full-res "
        "and a max-600px (longer side) copy for FLAC embed. Do not run this on archival MusicBrainz JPEGs."
    )
    p.add_argument("input", type=Path, help="Straightened, cropped, WhiBal'd front JPEG")
    p.add_argument("-o", "--output", type=Path, help="Full-res JPEG (default: *-descreen.jpg)")
    p.add_argument("--dpi", type=float, default=0.0, help="Override scan DPI (default: from file, else 600)")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality (default 90, 4:4:4 chroma)")
    p.add_argument("--strength", type=float, default=1.0, help="0–1 global blend toward descreened (default 1)")
    p.add_argument("--atten", type=float, default=0.85, help="Luma lattice attenuation 0–1 (default 0.85)")
    p.add_argument("--notch-lpi", type=float, default=18.0, help="Notch width in lines/inch (default 18)")
    p.add_argument("--tile", type=int, default=0, help="Unused; kept for compatibility (whole-image FFT)")
    p.add_argument("--smooth", type=float, default=0.0, help="Gated luma leftover-dot flatten (default 0)")
    p.add_argument("--chroma", type=float, default=0.70, help="Gated chroma rosette smooth (default 0.70)")
    p.add_argument(
        "--chroma-sig",
        type=float,
        default=1.8,
        help="Chroma rosette blur radius in screen periods (default 1.8)",
    )
    p.add_argument(
        "--ring",
        type=float,
        default=0.0,
        help="Outside-ink FFT ring suppress 0–1 (default 0; can mottle dense text)",
    )
    p.add_argument(
        "--edge-ring-y",
        type=float,
        default=0.0,
        help="Legacy luma edge-band blur 0–1 (default 0; smears ink — prefer --ring)",
    )
    p.add_argument(
        "--edge-ring-c",
        type=float,
        default=0.0,
        help="Legacy chroma edge-band blur 0–1 (default 0; smears ink — prefer --ring)",
    )
    p.add_argument(
        "--edge-ring-blur",
        type=float,
        default=3.0,
        help="Legacy edge-band width for --edge-ring-* (default 3)",
    )
    p.add_argument("--sharpen", type=float, default=0.08, help="Mild unsharp after descreen (default 0.08)")
    p.add_argument("--median", type=float, default=0.70, help="Gated luma median mix to break hex (default 0.70)")
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON object on stdout (for the Mac app / tooling)",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    src = args.input.expanduser().resolve()
    if not src.is_file():
        msg = f"not found: {src}"
        if args.json:
            print(json.dumps({"ok": False, "status": "error", "message": msg}))
        else:
            print(msg, file=sys.stderr)
        return 1

    rgb, meta = load_rgb(src)
    dpi = args.dpi or float(meta["dpi"][0] or 600.0) or 600.0

    full, info = descreen_rgb(
        rgb,
        dpi=dpi,
        atten=args.atten,
        notch_lpi=args.notch_lpi,
        tile=args.tile,
        smooth=args.smooth,
        sharpen=args.sharpen,
        strength=args.strength,
        chroma=args.chroma,
        median=args.median,
        chroma_sig=args.chroma_sig,
        ring=args.ring,
        edge_ring_y=args.edge_ring_y,
        edge_ring_c=args.edge_ring_c,
        edge_ring_blur=args.edge_ring_blur,
    )

    dest = (args.output or default_output(src)).expanduser().resolve()
    save_jpeg(dest, full, quality=args.quality, icc_profile=meta["icc_profile"], dpi=meta["dpi"])

    embed = dest.with_name(dest.stem + "-600.jpg")  # max 600 on longer side
    embed_rgb = resize_max(full, EMBED_MAX)
    save_jpeg(embed, embed_rgb, quality=args.quality, icc_profile=meta["icc_profile"], dpi=(72.0, 72.0))

    angles = ", ".join(f"{a:.0f}°" for a in info["angles"])
    status = f"{info['lpi']:.0f} LPI @ {angles}" if info["ok"] else f"{info['lpi']:.0f} LPI (fallback)"
    message = (
        f"{src.name}: {status}  screen~{info.get('screen_mean', 0):.2f} "
        f"→ {dest.name} ({full.shape[1]}×{full.shape[0]})"
    )
    if args.json:
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "descreened",
                    "message": message,
                    "inputPath": str(src),
                    "outputPath": str(dest),
                    "embedPath": str(embed),
                    "width": int(full.shape[1]),
                    "height": int(full.shape[0]),
                    "embedWidth": int(embed_rgb.shape[1]),
                    "embedHeight": int(embed_rgb.shape[0]),
                    "lpi": float(info["lpi"]),
                    "angles": [float(a) for a in info["angles"]],
                    "screenMean": float(info.get("screen_mean", 0.0)),
                    "detected": bool(info["ok"]),
                }
            )
        )
    else:
        print(message)
        print(f"  embed: {embed.name} ({embed_rgb.shape[1]}×{embed_rgb.shape[0]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
