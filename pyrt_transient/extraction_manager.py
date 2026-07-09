import os
import numpy as np
from astropy.table import Table
import astropy.wcs
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from sklearn.neighbors import KDTree

@dataclass
class ImageQuality:
    """Stores image quality metrics"""
    seeing: float  # FWHM in arcsec
    limiting_mag: float
    n_sources: int
    center_dist: float  # Distance of image center from field center

class ImageExtractionManager:
    """Manages multiple image extractions and selects optimal reference frame."""
    
    def __init__(self, detection_tables: List[Table]):
        """Initialize with list of detection tables.
        
        Args:
            detection_tables: List of detection tables with WCS metadata
        """
        self.detection_tables = detection_tables
        self.field_center = self._compute_field_center()
       # self.quality_metrics = self._compute_quality_metrics()
        #self.reference_idx = self._select_reference_image()
        
    def _compute_field_center(self) -> Tuple[float, float]:
        """Compute median center of all images."""
        ras = []
        decs = []
        for det in self.detection_tables:
            if hasattr(det,'meta'):
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
                #good_sources = det[np.all(det['MAGERR_AUTO'] < 0.2,isinstance(det['MAGERR_AUTO'], float))]
                #limiting_mag = np.percentile(good_sources['MAG_AUTO'], 90) if len(good_sources) > 0 else 0
            else:
                limiting_mag = 0
                
            # Count reliable sources
            n_sources = len(det)
            
            # Compute distance from field center
            img_ra = det.meta.get('CTRRA', det.meta.get('CRVAL1'))
            img_dec = det.meta.get('CTRDEC', det.meta.get('CRVAL2'))
            center_dist = np.sqrt((img_ra - ra_center)**2 + (img_dec - dec_center)**2)
            
            metrics.append(ImageQuality(
                seeing=seeing,
                limiting_mag=limiting_mag,
                n_sources=n_sources,
                center_dist=center_dist
            ))
            
        return metrics
    
    def _select_reference_image(self) -> int:
        """Select best reference image based on quality metrics.
        
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
            
        return np.argmax(scores)
    def generate_images(self, mag_candidate=None):
        """Generate images for transient candidates."""
        for detections in self.detection_tables:
            plt.figure()
            plt.scatter(detections['X_IMAGE'], detections['Y_IMAGE'], s=10, c=detections['MAG_CALIB'], cmap='viridis')
            plt.colorbar(label='MAG_AUTO')
            plt.xlabel('X_IMAGE')
            plt.ylabel('Y_IMAGE')
            plt.title(detections.meta.get('FITSFILE', 'unknown'))

            # mates FIX
            # Extract just the filename from the FITS path and save to your desired location
            fits_filename = os.path.basename(detections.meta.get('FITSFILE', 'unknown'))
            # plt.savefig(f"/home/fnovotny/data/{fits_filename}.png")

            # plt.savefig(detections.meta.get('FITSFILE', 'unknown') + '.png')
            plt.figure()
            plt.hist(detections["MAG_CALIB"])
            if mag_candidate:
                plt.axvline(mag_candidate, color='red', linestyle='--', label='Candidate mag')
            plt.legend()
            # one more FIX:
            # plt.savefig(detections.meta.get('FITSFILE', 'unknown') + '_hist.png')
            #plt.savefig(f"/home/fnovotny/data/{fits_filename}_hist.png")

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
    
    def get_detection_matches(self, 
                            candidates: Table, 
                            match_radius: float = 5.0) -> Dict[int, List[int]]:
        """Find matching detections for each candidate in all images.
        
        Args:
            candidates: Table of candidates
            match_radius: Matching radius in pixels
            
        Returns:
            Dictionary mapping candidate index to list of detection indices per image
        """
        matches = {}
        
        for i, det in enumerate(self.detection_tables):
            # Get WCS for this image
            wcs = astropy.wcs.WCS(det.meta)
            
            # Transform candidate coordinates to this image
            x, y = wcs.all_world2pix(
                candidates['ALPHA_J2000'],
                candidates['DELTA_J2000'],
                1
            )
            
            # Create KDTree for detections
            det_coords = np.column_stack((det['X_IMAGE'], det['Y_IMAGE']))
            tree = KDTree(det_coords)
            
            # Find matches
            cand_coords = np.column_stack((x, y))
            indices = tree.query_radius(cand_coords, r=match_radius)
            
            # Store matches
            for cand_idx, det_indices in enumerate(indices):
                if cand_idx not in matches:
                    matches[cand_idx] = {}
                matches[cand_idx][i] = det_indices.tolist()
                
        return matches
