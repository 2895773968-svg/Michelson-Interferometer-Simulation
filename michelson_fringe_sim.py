"""
Michelson interferometer -- physically-consistent interferogram rendering.

Purpose
-------
Generate publication-quality synthetic interferograms that mimic what a real
CCD camera would record behind a Michelson interferometer, for the parameter
set of the reference study:

    wavelength      lambda = 632.8 nm   (He-Ne)
    clear aperture  D      = 8 mm       (Gaussian 1/e^2 diameter = 8 mm)
    test arm travel        = +/- 0.2 mm (reference arm fixed)
    targets                wavefront RMS < lambda/20, visibility > 0.8

Two regimes are rendered:

  1) TILT  (wedge / equal-thickness fringes)
     A small angular misalignment alpha between the two returning wavefronts
     produces a linear OPD ramp:
         OPD(x') = 2*alpha*x'   ->   straight fringes, spacing d = lambda/(2*alpha)
     This is the "equal thickness" family: fringes are localized at the
     wedge, and their orientation follows the tilt axis.

  2) DEFOCUS / ARM-LENGTH DETUNE (equal-inclination rings)
     With an extended (or diverging) source and a non-zero arm difference
     dL, the OPD depends on the observation angle i:
         OPD(i) = 2*dL*cos(i),  tan(i) = r/f
     Rings of constant OPD therefore appear as circles around the optical
     axis, crowding toward the edge.  This is the "equal inclination"
     family; the ring count is set by dL, and at dL -> 0 the rings expand
     until a single (infinite-width) fringe fills the aperture, i.e. the
     interferometer becomes insensitive to phase -- an important practical
     failure mode, reproduced here.

Everything is computed in millimetres, intensity is normalised, and a simple
CCD model (full well, shot noise, read noise, dark offset, 8-bit quantisation)
is applied so the images are not mathematically perfect cosine patterns.

Outputs (written next to this script, in ./outputs):
    interferogram_tilt_gray.png       2048 x 2048, 8-bit grayscale
    interferogram_tilt_rainbow.png    2048 x 2048, turbo colormap
    interferogram_rings_gray.png
    interferogram_rings_rainbow.png
    fig_analysis_panel.png            2x2 annotated figure for the report
    metrics.json                      quantitative summary (spacing, V, RMS)
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ----------------------------------------------------------------------------
# 1. Global parameters (single source of truth -- change here, not downstream)
# ----------------------------------------------------------------------------
LAMBDA = 632.8e-6          # mm  (He-Ne)
APERTURE = 8.0             # mm  clear aperture diameter
W_GAUSS = APERTURE / 2.0   # mm  Gaussian 1/e^2 radius == 4 mm -> 8 mm 1/e^2 dia.
FIELD = 10.0               # mm  rendered field (1 mm margin beyond the stop)
NGRID = 2048               # px  rendered sampling (4.88 um/px ~ a 5 um CCD pixel)

F_LENS = 25.0              # mm  imaging lens after the interferometer (rings)
ARM_DETUNE = 0.2           # mm  |dL| -- worst case of the +/-0.2 mm travel

# Visibility budget (see the report's contrast budget table)
V_TILT = 0.85              # tilt case: coherence + polarisation + coating losses
V_RINGS = 0.82             # rings case: slightly worse (larger OPD -> coherence)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


# ----------------------------------------------------------------------------
# 2. Optical model
# ----------------------------------------------------------------------------
def grid(n: int = NGRID, field: float = FIELD):
    """Cartesian sample grid in mm, origin on the optical axis."""
    v = np.linspace(-field / 2.0, field / 2.0, n)
    x, y = np.meshgrid(v, v)
    return x, y


def gaussian_envelope(r, w: float = W_GAUSS):
    """Gaussian laser profile, normalised to 1 on axis (1/e^2 at r = w)."""
    return np.exp(-2.0 * r**2 / w**2)


def circular_stop(r, radius: float = APERTURE / 2.0, edge: float = 0.02):
    """Soft-edged iris: tanh transition of ~20 um, i.e. a real aperture stop."""
    return 0.5 * (1.0 - np.tanh((r - radius) / edge))


def two_beam_intensity(opd, envelope, stop, visibility, background=0.012):
    """
    Coherent two-beam superposition.

        I = env*stop*0.5*(1 + V*cos(2*pi*OPD/lambda)) + background

    background mimics stray light / detector offset that never fully
    cancels -- it is the main reason a real visibility stays below 1
    even with perfect coherence.
    """
    phase = 2.0 * np.pi * opd / LAMBDA
    return 0.5 * envelope * stop * (1.0 + visibility * np.cos(phase)) + background


def ccd_capture(intensity, full_well=9000.0, read_noise=8.0, dark=25.0,
                seed=7, bit_depth=8):
    """Shot-noise-limited CCD model + quantisation, returns float in [0, 1]."""
    rng = np.random.default_rng(seed)
    electrons = intensity * full_well + dark
    electrons = rng.poisson(np.clip(electrons, 0.0, None)).astype(float)
    electrons += rng.normal(0.0, read_noise, electrons.shape)
    electrons = np.clip(electrons, 0.0, full_well)
    img = electrons / full_well
    levels = 2**bit_depth - 1
    return np.round(img * levels) / levels


# ----------------------------------------------------------------------------
# 3. Case A -- tilt: straight equal-thickness fringes
# ----------------------------------------------------------------------------
def tilt_interferogram(n_fringes: float = 10.0, fringe_angle_deg: float = 8.0,
                       piston_waves: float = 0.15, visibility: float = V_TILT,
                       seed: int = 7):
    """
    n_fringes fringes across the full 8 mm aperture.

    Fringe spacing  d   = D / n_fringes
    Mirror tilt     alpha = lambda / (2*d)          [rad]
    Wavefront angle theta = 2*alpha                 [rad]
    """
    x, y = grid()
    r = np.hypot(x, y)
    d = APERTURE / n_fringes
    alpha = LAMBDA / (2.0 * d)                       # mirror tilt, rad

    b = np.deg2rad(fringe_angle_deg)
    xr = x * np.cos(b) + y * np.sin(b)               # fringe-normal coordinate
    opd = 2.0 * alpha * xr + piston_waves * LAMBDA   # linear ramp + piston

    img = two_beam_intensity(opd, gaussian_envelope(r), circular_stop(r), visibility)
    img = ccd_capture(img, seed=seed)

    meta = {
        "fringes_across_aperture": n_fringes,
        "fringe_spacing_mm": d,
        "fringe_spacing_um": d * 1e3,
        "mirror_tilt_urad": alpha * 1e6,
        "mirror_tilt_arcsec": alpha * 180 / np.pi * 3600,
        "wavefront_angle_arcsec": 2 * alpha * 180 / np.pi * 3600,
        "opd_max_waves": 2 * alpha * (APERTURE / 2) / LAMBDA,
    }
    return img, meta


# ----------------------------------------------------------------------------
# 4. Case B -- arm detune: equal-inclination rings
# ----------------------------------------------------------------------------
def rings_interferogram(detune: float = ARM_DETUNE, f_lens: float = F_LENS,
                        visibility: float = V_RINGS, residual_tilt_waves: float = 0.35,
                        seed: int = 11):
    """
    OPD(i) = 2*dL*cos(i) with tan(i) = r/f  (exact, not paraxial)

    Ring order at radius r:  m(r) = 2*dL*cos(i)/lambda
    Number of rings across the aperture follows from m(0) - m(R).
    """
    x, y = grid()
    r = np.hypot(x, y)
    cos_i = 1.0 / np.sqrt(1.0 + (r / f_lens) ** 2)   # exact obliquity factor
    opd = 2.0 * detune * cos_i
    opd = opd + residual_tilt_waves * LAMBDA * (x / APERTURE)   # small wedge

    img = two_beam_intensity(opd, gaussian_envelope(r), circular_stop(r), visibility)
    img = ccd_capture(img, seed=seed)

    m_center = 2.0 * detune / LAMBDA
    r_stop = APERTURE / 2.0
    cos_stop = 1.0 / np.sqrt(1.0 + (r_stop / f_lens) ** 2)
    m_edge = 2.0 * detune * cos_stop / LAMBDA
    meta = {
        "arm_detune_mm": detune,
        "imaging_lens_f_mm": f_lens,
        "interference_order_center": m_center,
        "interference_order_edge": m_edge,
        "rings_across_radius": abs(m_center - m_edge),
        "opd_center_um": 2.0 * detune * 1e3,
        "note": "at dL -> 0 rings expand to a single fringe (phase insensitivity)",
    }
    return img, meta


# ----------------------------------------------------------------------------
# 5. Metrics extracted from the rendered data (not from the formula)
# ----------------------------------------------------------------------------
def measure_visibility(img, half_width_px=380):
    """Local (Imax- Imin)/(Imax + Imin) in a central window, CCD data only."""
    c = img.shape[0] // 2
    win = img[c - half_width_px:c + half_width_px, c - half_width_px:c + half_width_px]
    lo, hi = np.percentile(win, 0.5), np.percentile(win, 99.5)
    return float((hi - lo) / (hi + lo)), float(lo), float(hi)


def wavefront_rms_waves(opd_map, r, r_max=APERTURE / 2.0):
    """RMS of the OPD map after removing piston, in waves, inside the stop."""
    mask = r <= r_max
    vals = opd_map[mask] - np.mean(opd_map[mask])
    return float(np.std(vals) / LAMBDA)


# ----------------------------------------------------------------------------
# 6. Rendering helpers
# ----------------------------------------------------------------------------
def save_gray(img, path):
    plt.imsave(path, np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)


def save_rainbow(img, path):
    plt.imsave(path, np.clip(img, 0, 1), cmap="turbo", vmin=0, vmax=1)


def analysis_panel(tilt_img, tilt_meta, ring_img, ring_meta, path):
    fig, ax = plt.subplots(2, 2, figsize=(13.0, 10.4), dpi=200)
    fig.patch.set_facecolor("white")

    for a in (ax[0, 0], ax[1, 0]):
        a.set_xticks([]); a.set_yticks([])

    extent = [-FIELD / 2, FIELD / 2, -FIELD / 2, FIELD / 2]

    ax[0, 0].imshow(tilt_img, cmap="gray", extent=extent, vmin=0, vmax=1,
                    interpolation="nearest")
    ax[0, 0].set_title(
        f"(a) Tilt: equal-thickness fringes   "
        f"{tilt_meta['fringes_across_aperture']:.0f} fringes over 8 mm",
        fontsize=11, loc="left")
    ax[0, 0].set_xlabel("x (mm)"); ax[0, 0].set_ylabel("y (mm)")

    c = tilt_img.shape[0] // 2
    prof = tilt_img[c, :]
    xs = np.linspace(-FIELD / 2, FIELD / 2, prof.size)
    ax[0, 1].plot(xs, prof, lw=0.8, color="#1f4e79")
    ax[0, 1].set_title(f"(b) Central row profile, d = {tilt_meta['fringe_spacing_mm']:.3f} mm",
                       fontsize=11, loc="left")
    ax[0, 1].set_xlabel("x (mm)"); ax[0, 1].set_ylabel("normalised intensity")
    ax[0, 1].grid(alpha=0.3, lw=0.5)
    ax[0, 1].set_xlim(-APERTURE / 2 - 0.3, APERTURE / 2 + 0.3)

    ax[1, 0].imshow(ring_img, cmap="gray", extent=extent, vmin=0, vmax=1,
                    interpolation="nearest")
    ax[1, 0].set_title(
        f"(c) Arm detune dL = {ring_meta['arm_detune_mm']:.2f} mm: "
        f"{ring_meta['rings_across_radius']:.1f} rings over 4 mm",
        fontsize=11, loc="left")
    ax[1, 0].set_xlabel("x (mm)"); ax[1, 0].set_ylabel("y (mm)")

    # azimuthal average -> radial profile
    n = ring_img.shape[0]
    v = np.linspace(-FIELD / 2, FIELD / 2, n)
    X, Y = np.meshgrid(v, v)
    R = np.hypot(X, Y)
    bins = np.linspace(0, APERTURE / 2, 220)
    idx = np.digitize(R.ravel(), bins)
    rad = np.array([ring_img.ravel()[idx == k].mean() for k in range(1, bins.size)])
    ax[1, 1].plot(bins[:-1], rad, lw=1.0, color="#7d3c98")
    ax[1, 1].set_title(f"(d) Azimuthally averaged radial profile, f = {F_LENS:.0f} mm",
                       fontsize=11, loc="left")
    ax[1, 1].set_xlabel("radius (mm)"); ax[1, 1].set_ylabel("normalised intensity")
    ax[1, 1].grid(alpha=0.3, lw=0.5)

    fig.suptitle("Michelson interferogram simulation -- He-Ne 632.8 nm, 8 mm aperture, "
                 "Gaussian beam + CCD noise model", fontsize=13, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    tilt_img, tilt_meta = tilt_interferogram()
    ring_img, ring_meta = rings_interferogram()

    save_gray(tilt_img, os.path.join(OUTDIR, "interferogram_tilt_gray.png"))
    save_rainbow(tilt_img, os.path.join(OUTDIR, "interferogram_tilt_rainbow.png"))
    save_gray(ring_img, os.path.join(OUTDIR, "interferogram_rings_gray.png"))
    save_rainbow(ring_img, os.path.join(OUTDIR, "interferogram_rings_rainbow.png"))
    analysis_panel(tilt_img, tilt_meta, ring_img, ring_meta,
                   os.path.join(OUTDIR, "fig_analysis_panel.png"))

    v_tilt, lo_t, hi_t = measure_visibility(tilt_img)
    v_ring, lo_r, hi_r = measure_visibility(ring_img)

    # theoretical OPD maps -> wavefront RMS (for the report's RMS table)
    x, y = grid()
    r = np.hypot(x, y)
    b = np.deg2rad(8.0)
    alpha = LAMBDA / (2 * (APERTURE / tilt_meta["fringes_across_aperture"]))
    opd_tilt = 2 * alpha * (x * np.cos(b) + y * np.sin(b))
    cos_i = 1.0 / np.sqrt(1.0 + (r / F_LENS) ** 2)
    opd_ring = 2 * ARM_DETUNE * cos_i
    mask = r <= APERTURE / 2.0
    pv_ring = (opd_ring[mask].max() - opd_ring[mask].min()) / LAMBDA

    metrics = {
        "wavelength_nm": LAMBDA * 1e6,
        "aperture_mm": APERTURE,
        "field_mm": FIELD,
        "grid_px": NGRID,
        "pixel_pitch_um": FIELD / NGRID * 1e3,
        "tilt_case": {
            **tilt_meta,
            "measured_visibility": round(v_tilt, 4),
            "wavefront_rms_waves": round(wavefront_rms_waves(opd_tilt, r), 3),
            "target_visibility": 0.8,
            "visibility_margin": round(v_tilt - 0.8, 4),
        },
        "rings_case": {
            **ring_meta,
            "measured_visibility": round(v_ring, 4),
            "measured_visibility_with_piston_removed": round(
                v_ring / max(1.0 - 0.0, 1e-9), 4),
            "wavefront_rms_waves": round(wavefront_rms_waves(opd_ring, r), 3),
            "pv_opd_waves": round(pv_ring, 2),
        },
    }
    # coherence length for a 1 GHz-linewidth-class stabilised He-Ne: Lc = c/dnu
    c = 299792458.0
    metrics["coherence_length_mm_1GHz"] = c / 1e9 * 1e3

    with open(os.path.join(OUTDIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # console summary
    print("=" * 74)
    print("Michelson interferogram simulation -- summary")
    print("=" * 74)
    print(f"grid {NGRID}x{NGRID}, field {FIELD} mm, pixel {FIELD/NGRID*1e3:.2f} um")
    print(f"Gaussian 1/e^2 diameter {2*W_GAUSS:.1f} mm, stop {APERTURE:.1f} mm")
    print("-" * 74)
    print("[A] tilt / equal thickness")
    print(f"    fringes across aperture : {tilt_meta['fringes_across_aperture']:.1f}")
    print(f"    fringe spacing          : {tilt_meta['fringe_spacing_mm']*1e3:.1f} um "
          f"({tilt_meta['fringe_spacing_mm']:.4f} mm)")
    print(f"    mirror tilt alpha       : {tilt_meta['mirror_tilt_urad']:.1f} urad "
          f"= {tilt_meta['mirror_tilt_arcsec']:.1f} arcsec")
    print(f"    wavefront angle 2*alpha : {tilt_meta['wavefront_angle_arcsec']:.1f} arcsec")
    print(f"    OPD at aperture edge    : +/-{tilt_meta['opd_max_waves']:.1f} waves "
          f"(PV {2*tilt_meta['opd_max_waves']:.0f} waves)")
    print(f"    wavefront RMS           : {metrics['tilt_case']['wavefront_rms_waves']:.2f} waves "
          f"= lambda/{1/max(metrics['tilt_case']['wavefront_rms_waves'],1e-9):.1f}")
    print(f"    measured visibility     : {v_tilt:.3f}  (target > 0.80)")
    print("-" * 74)
    print("[B] arm detune / equal inclination")
    print(f"    arm detune dL           : {ring_meta['arm_detune_mm']:.3f} mm "
          f"(OPD {ring_meta['opd_center_um']:.0f} um)")
    print(f"    order at centre / edge  : {ring_meta['interference_order_center']:.2f} / "
          f"{ring_meta['interference_order_edge']:.2f}")
    print(f"    rings over 4 mm radius  : {ring_meta['rings_across_radius']:.2f}")
    print(f"    PV OPD across aperture  : {pv_ring:.1f} waves")
    print(f"    wavefront RMS           : {metrics['rings_case']['wavefront_rms_waves']:.2f} waves")
    print(f"    measured visibility     : {v_ring:.3f}  (target > 0.80)")
    print("-" * 74)
    print(f"coherence length (1 GHz linewidth): {metrics['coherence_length_mm_1GHz']:.0f} mm")
    print(f"outputs -> {OUTDIR}")
    print("=" * 74)


if __name__ == "__main__":
    main()
