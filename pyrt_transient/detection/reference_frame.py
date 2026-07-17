"""ReferenceFrameSelector -- a genuine bug fix, not a refactor.
extraction_manager.py's ImageExtractionManager never actually sets
self.reference_idx (_select_reference_image() is commented out in
__init__), so its transform_to_reference/validate_reference_coordinates/
get_detection_matches have always been broken -- confirmed via grep that
none of them (nor generate_images) have any call sites anywhere; only
.field_center is ever used (by pipeline_magic.py).

Built fresh here rather than patching extraction_manager.py in place, since
nothing depends on the old broken behavior -- extraction_manager.py itself
is untouched, still providing .field_center for the live production path.

generate_images is dropped entirely: dead-end plotting method, commented-out
savefig calls, and a mag_candidate parameter only used for one axvline --
confirmed no call sites.

get_detection_matches' KDTree block is replaced with core/matching.match_radius
(coord_system="pixel") -- this was the fourth independent implementation of
the same radius-match logic in this codebase.

No check_baseline.py coverage here (this code was never callable before, so
there's no existing behavior to preserve) -- see tests/test_reference_frame.py
for fresh unit tests against synthetic WCS headers instead.

Target-aware selection (target_ra/target_dec) added when this became the
own-epoch template picker for the subtraction detection strategy (see
detection/subtraction/templates.py, get_template_own_epoch): picking a
reference epoch purely on generic image quality (seeing/depth/source count)
has no way to know whether the *target itself* already has real flux in a
candidate reference epoch. For a genuinely new transient that's harmless --
there's nothing there yet in any epoch to accidentally subtract away. But
for continued monitoring of an already-known, slowly-evolving source (the
common case once a target has been flagged and a campaign starts), it can
silently pick a reference epoch where the target is already near its normal
brightness, and difference against it -- which reveals only the (possibly
tiny) change relative to that reference, not the target's true magnitude,
with no warning that this happened. Confirmed directly against the real
tests/2026kid/ campaign: AT2026kid's own science-image magnitude was flat
(15.35-15.73 mag, no real trend) across all seven nights, so *any* pair of
those nights used as science/reference would reproduce the same
under-representation this analysis found for the 0425->0426 pair
specifically (true mag ~15.65, own-epoch residual mag 17.5-18.1 depending
on differencing engine -- see FUTURE_IDEAS.md).

target_ra/target_dec, when provided, make epoch selection prefer a
genuinely target-free reference epoch when one exists, and log a clear
warning (rather than silently degrading) when it doesn't -- e.g. exactly
the all-epochs-contaminated case a persistent, already-known transient like
AT2026kid produces.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import astropy.wcs
import numpy as np
from astropy.table import Table

from pyrt_transient.core.matching import match_radius

logger = logging.getLogger("detection.reference_frame")


@dataclass
class ImageQuality:
    """Stores image quality metrics"""
    seeing: float  # FWHM in arcsec
    limiting_mag: float
    n_sources: int
    center_dist: float  # Distance of image center from field center
    target_present: bool = False  # True if a source is detected near the known target position


class ReferenceFrameSelector:
    """Manages multiple image extractions and selects optimal reference frame."""

    def __init__(
        self,
        detection_tables: List[Table],
        target_ra: Optional[float] = None,
        target_dec: Optional[float] = None,
        target_exclusion_radius_arcsec: float = 5.0,
    ):
        """Initialize with list of detection tables.

        Args:
            detection_tables: List of detection tables with WCS metadata
            target_ra/target_dec: Known target position, optional. When
                given, epoch selection avoids picking a reference epoch
                where a source already exists at this position (see this
                module's docstring for why) -- unset (the default)
                preserves the exact original generic-quality-only
                selection behavior.
            target_exclusion_radius_arcsec: Match radius around
                target_ra/target_dec used to decide "is the target present
                in this epoch".
        """
        self.detection_tables = detection_tables
        self.target_ra = target_ra
        self.target_dec = target_dec
        self.target_exclusion_radius_arcsec = target_exclusion_radius_arcsec
        self.field_center = self._compute_field_center()
        self.quality_metrics = self._compute_quality_metrics()
        self.reference_idx = self._select_reference_image()

    def _compute_field_center(self) -> Tuple[float, float]:
        """Compute median center of all images."""
        ras = []
        decs = []
        for det in self.detection_tables:
            if hasattr(det, 'meta'):
                ras.append(det.meta.get('CTRRA', det.meta.get('CRVAL1')))
                decs.append(det.meta.get('CTRDEC', det.meta.get('CRVAL2')))
        return np.median(ras), np.median(decs)

    def _compute_quality_metrics(self) -> List[ImageQuality]:
        """Compute quality metrics for each image."""
        metrics = []
        ra_center, dec_center = self.field_center

        for det in self.detection_tables:
            # Get basic metrics
            seeing = det.meta.get('FWHM', float('inf'))
            # Compute limiting magnitude from faintest reliable detections
            if 'MAG_AUTO' in det.columns and 'MAGERR_AUTO' in det.columns:
                # Valid finite values with MAGERR_AUTO < 0.2
                valid_mask = np.isfinite(det['MAGERR_AUTO']) & (det['MAGERR_AUTO'] < 0.2)
                good_sources = det[valid_mask]
                limiting_mag = np.percentile(good_sources['MAG_AUTO'], 90) if len(good_sources) > 0 else 0
            else:
                limiting_mag = 0

            # Count reliable sources
            n_sources = len(det)

            # Compute distance from field center
            img_ra = det.meta.get('CTRRA', det.meta.get('CRVAL1'))
            img_dec = det.meta.get('CTRDEC', det.meta.get('CRVAL2'))
            center_dist = np.sqrt((img_ra - ra_center) ** 2 + (img_dec - dec_center) ** 2)

            target_present = self._is_target_present(det) if self.target_ra is not None else False

            metrics.append(ImageQuality(
                seeing=seeing,
                limiting_mag=limiting_mag,
                n_sources=n_sources,
                center_dist=center_dist,
                target_present=target_present,
            ))

        return metrics

    def _is_target_present(self, det: Table) -> bool:
        """Whether a detection exists within target_exclusion_radius_arcsec
        of the known target position in this epoch's table -- see this
        module's docstring for why an own-epoch template picker needs to
        know this.
        """
        if 'ALPHA_J2000' not in det.colnames or 'DELTA_J2000' not in det.colnames or len(det) == 0:
            return False
        ra = np.asarray(det['ALPHA_J2000'], dtype=float)
        dec = np.asarray(det['DELTA_J2000'], dtype=float)
        dra = (ra - self.target_ra) * np.cos(np.radians(self.target_dec))
        ddec = dec - self.target_dec
        sep_arcsec = np.sqrt(dra ** 2 + ddec ** 2) * 3600.0
        return bool(np.any(sep_arcsec <= self.target_exclusion_radius_arcsec))

    def _select_reference_image(self) -> int:
        """Select best reference image based on quality metrics.

        When target_ra/target_dec was given: restricts the choice to
        epochs where the target isn't already present, if any exist, and
        only falls back to the single best-quality epoch overall (with a
        clear warning, not a silent one) when every candidate epoch already
        has the target in it -- e.g. the AT2026kid case in this module's
        docstring, where the target is essentially always present.

        Returns:
            Index of best reference image
        """
        scores = []
        for quality in self.quality_metrics:
            # Compute score where higher is better
            score = (
                (1.0 / quality.seeing) * 0.4 +  # Better seeing
                (quality.limiting_mag / 20.0) * 0.2 +  # Deeper image
                (quality.n_sources / 1000.0) * 0.1 +  # More sources
                (1.0 / (1.0 + quality.center_dist)) * 0.1  # Closer to field center
            )
            scores.append(score)

        if self.target_ra is not None:
            clean_indices = [i for i, q in enumerate(self.quality_metrics) if not q.target_present]
            if clean_indices:
                return max(clean_indices, key=lambda i: scores[i])
            logger.warning(
                "ReferenceFrameSelector: target present in every candidate epoch -- "
                "no genuinely quiescent own-epoch template available. Falling back to "
                "the best-quality epoch overall, but differencing against it will "
                "under-represent the target's true brightness (it will reveal only "
                "the change relative to that epoch, not the total magnitude -- see "
                "FUTURE_IDEAS.md's own-epoch-vs-PS1 analysis). Consider "
                "template_source='ps1'/'legacysurvey' instead for absolute photometry."
            )

        return int(np.argmax(scores))

    def transform_to_reference(self, candidates: Table) -> Table:
        """Transform candidate coordinates to reference image system.

        Args:
            candidates: Table of candidates with ALPHA_J2000 and DELTA_J2000 columns

        Returns:
            Table with added X_REF and Y_REF columns in reference image coordinates
        """
        # Get reference image WCS
        ref_det = self.detection_tables[self.reference_idx]
        ref_wcs = astropy.wcs.WCS(ref_det.meta)

        # Transform coordinates
        x_ref, y_ref = ref_wcs.all_world2pix(
            candidates['ALPHA_J2000'],
            candidates['DELTA_J2000'],
            1
        )

        # Add reference coordinates to table
        result = candidates.copy()
        result['X_REF'] = x_ref
        result['Y_REF'] = y_ref

        # Add reference image metadata
        result.meta['reference_image'] = ref_det.meta.get('FITSFILE', 'unknown')
        result.meta['reference_idx'] = self.reference_idx

        return result

    def validate_reference_coordinates(self, candidates: Table, margin: float = 10.0) -> Table:
        """Filter candidates to those with valid reference coordinates.

        Args:
            candidates: Candidate table with X_REF and Y_REF columns
            margin: Allowed margin outside image bounds in pixels

        Returns:
            Filtered candidate table
        """
        ref_det = self.detection_tables[self.reference_idx]
        width = ref_det.meta.get('NAXIS1', ref_det.meta.get('IMAGEW'))
        height = ref_det.meta.get('NAXIS2', ref_det.meta.get('IMAGEH'))

        if width is None or height is None:
            raise ValueError("Cannot determine reference image dimensions")

        # Create mask for valid coordinates
        valid = (
            (candidates['X_REF'] >= -margin) &
            (candidates['X_REF'] < width + margin) &
            (candidates['Y_REF'] >= -margin) &
            (candidates['Y_REF'] < height + margin)
        )

        return candidates[valid]

    def get_detection_matches(
        self,
        candidates: Table,
        match_radius_px: float = 5.0,
    ) -> Dict[int, Dict[int, List[int]]]:
        """Find matching detections for each candidate in all images.

        Args:
            candidates: Table of candidates
            match_radius_px: Matching radius in pixels

        Returns:
            Dictionary mapping candidate index -> {image index: [detection indices]}
        """
        matches: Dict[int, Dict[int, List[int]]] = {}
        n_candidates = len(candidates)

        for i, det in enumerate(self.detection_tables):
            # Transform candidate coordinates to this image
            wcs = astropy.wcs.WCS(det.meta)
            x, y = wcs.all_world2pix(
                candidates['ALPHA_J2000'],
                candidates['DELTA_J2000'],
                1
            )
            cand_coords = np.column_stack((x, y))
            det_coords = np.column_stack((det['X_IMAGE'], det['Y_IMAGE']))

            per_cand_matches: List[List[int]] = [[] for _ in range(n_candidates)]
            if n_candidates > 0 and len(det) > 0:
                idx_a, idx_b, _ = match_radius(
                    cand_coords, det_coords, match_radius_px, coord_system="pixel"
                )
                for a, b in zip(idx_a, idx_b):
                    per_cand_matches[int(a)].append(int(b))

            for cand_idx in range(n_candidates):
                matches.setdefault(cand_idx, {})[i] = per_cand_matches[cand_idx]

        return matches
