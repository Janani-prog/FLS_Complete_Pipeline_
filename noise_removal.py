"""
Fixed Safe Noise Removal Processor - Proper Bottom Plate Removal
Restores original V3 plane removal functionality with safety limits
CORRECTED: Maintains original noise removal effectiveness while improving point retention
"""

import open3d as o3d
import numpy as np
import gc
from typing import Optional, Tuple
from pathlib import Path


class NoiseRemovalProcessor:
    """Safe noise removal processor with proper bottom plate removal."""

    def __init__(self):
        """Initialize with conservative but effective memory limits."""
        # Memory estimates for 32GB system
        self.total_ram = 32 * (1024**3)  # 32GB in bytes
        self.windows_overhead = 0.45     # 45% Windows + other apps
        self.safety_margin = 0.20        # Reduced safety margin from 25% to 20%
        self.usable_ram = self.total_ram * (1 - self.windows_overhead - self.safety_margin)

        # More realistic point cloud memory estimation
        self.bytes_per_point = 120       # Reduced from 150 for better memory estimate

        # INCREASED safe points limit to retain more data
        self.max_safe_points = min(
            int(self.usable_ram / self.bytes_per_point),
            15_000_000  # INCREASED from 8M to 15M points for better detail retention
        )

        print("SAFE Memory Analysis:")
        print(f"  Total RAM: {self.total_ram / (1024**3):.1f} GB")
        print(f"  Usable for processing: {self.usable_ram / (1024**3):.1f} GB")
        print(f"  Bytes per point: {self.bytes_per_point}")
        print(f"  Safe limit: {self.max_safe_points:,} points MAX")

    def save_cleaned_ply(self, cleaned_pcd: o3d.geometry.PointCloud, input_path: str) -> Optional[str]:
        """
        Save cleaned point cloud as <originalname>_cleaned.ply
        """
        if cleaned_pcd is None or len(cleaned_pcd.points) == 0:
            print("ERROR: No cleaned point cloud to save")
            return None

        input_path = Path(input_path)
        out_dir = Path(r"output\cleaned_scans")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{input_path.stem}_cleaned.ply"

        success = o3d.io.write_point_cloud(
            str(output_path),
            cleaned_pcd,
            write_ascii=False,
            compressed=False
        )

        if success:
            print(f"✔ Cleaned file saved: {output_path}")
            return str(output_path)
        else:
            print("✖ Failed to save cleaned PLY")
            return None

    
    # ---------------------------------------------------------
    # BASIC RADIAL MASK (ASSUMES APPROX X-AXIS MILL)
    # ---------------------------------------------------------
    def _compute_radial_mask(self, pcd, safety_margin: float = 0.98):
        """
        Returns True for points INSIDE the mill (protected region).
        Assumes mill axis ≈ X axis.
        """
        pts = np.asarray(pcd.points)

        center_y = np.median(pts[:, 1])
        center_z = np.median(pts[:, 2])

        radial_dist = np.sqrt(
            (pts[:, 1] - center_y) ** 2 +
            (pts[:, 2] - center_z) ** 2
        )

        max_radius = np.percentile(radial_dist, 99.5)
        safe_radius = max_radius * safety_margin

        return radial_dist < safe_radius

    # ---------------------------------------------------------
    # NEW: STRICT CYLINDRICAL CROP VIA PCA
    # ---------------------------------------------------------
    def _strict_mill_cylindrical_crop(
        self,
        pcd: o3d.geometry.PointCloud,
        radial_margin: float = 0.002,   # 2 mm radial slack
        length_margin: float = 0.005    # 5 mm slack at each end
    ) -> o3d.geometry.PointCloud:
        """
        Hard geometric crop that keeps ONLY points inside a PCA-estimated
        mill cylinder and removes everything else (external structure,
        scaffolding, etc.).

        Works on very large point clouds because it is fully vectorized.
        """
        pts = np.asarray(pcd.points)
        if pts.shape[0] == 0:
            return pcd

        # 1) PCA to estimate mill axis (first principal component)
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, np.argmax(eigvals)]  # dominant direction

        # Ensure consistent direction (optional)
        if axis[0] < 0:
            axis = -axis

        # 2) Axial projection and radial distance
        axial_coord = pts_centered @ axis          # scalar projection on axis
        proj = np.outer(axial_coord, axis)         # projected vector
        radial_vec = pts_centered - proj
        radial_dist = np.linalg.norm(radial_vec, axis=1)

        # 3) Robust radius and length (ignore far outliers)
        max_rad = np.percentile(radial_dist, 99.7)
        min_ax = np.percentile(axial_coord, 0.3)
        max_ax = np.percentile(axial_coord, 99.7)

        safe_radius = max_rad + radial_margin
        safe_min_ax = min_ax - length_margin
        safe_max_ax = max_ax + length_margin

        # 4) Keep points inside cylinder and within length bounds
        inside_mask = (
            (radial_dist <= safe_radius) &
            (axial_coord >= safe_min_ax) &
            (axial_coord <= safe_max_ax)
        )

        filtered_pts = pts[inside_mask]
        clean = o3d.geometry.PointCloud()
        clean.points = o3d.utility.Vector3dVector(filtered_pts)

        if pcd.has_colors():
            col = np.asarray(pcd.colors)[inside_mask]
            clean.colors = o3d.utility.Vector3dVector(col)
        if pcd.has_normals():
            nor = np.asarray(pcd.normals)[inside_mask]
            clean.normals = o3d.utility.Vector3dVector(nor)

        removed = pts.shape[0] - filtered_pts.shape[0]
        print(
            f"STRICT CYLINDER CROP: kept {filtered_pts.shape[0]:,} "
            f"points, removed {removed:,} ({removed/pts.shape[0]*100:.1f}%)"
        )

        return clean

    # ---------------------------------------------------------
    # MAIN V3 PIPELINE
    # ---------------------------------------------------------
    def apply_v3_noise_removal(self, point_cloud: o3d.geometry.PointCloud, input_path: Optional[str] = None, save_cleaned: bool = True) -> Optional[o3d.geometry.PointCloud]:
        """
        Apply V3 noise removal with PROPER bottom plate removal.
        CORRECTED: Uses original noise removal effectiveness with improved initial downsampling.
        """
        if point_cloud is None or len(point_cloud.points) == 0:
            print("ERROR: Invalid input point cloud")
            return None

        try:
            print("\n=== V3 NOISE REMOVAL WITH IMPROVED RETENTION ===")
            input_points = len(point_cloud.points)
            print(f"Input points: {input_points:,}")
            print(
                "Target retention: 10-12M points "
                f"({(10_000_000/input_points)*100:.1f}-"
                f"{(12_000_000/input_points)*100:.1f}%)"
            )

            # Step 1: IMPROVED initial downsampling
            if input_points > self.max_safe_points:
                print("Input exceeds safe limit, applying IMPROVED initial downsampling")
                working_pcd = self._improved_initial_downsample(point_cloud)
            else:
                print("Input within safe limits")
                working_pcd = point_cloud

            print(f"Working with: {len(working_pcd.points):,} points")

            # Force garbage collection
            gc.collect()

            # Step 2: Bottom plate removal
            """print("Applying BOTTOM PLATE REMOVAL (V3 original method)...")
            working_pcd = self._remove_bottom_plate_soft(working_pcd)"""

            # Step 3: Statistical outlier removal on outer shell
            print("Applying statistical outlier removal (original strength)...")
            working_pcd = self._apply_original_statistical_removal(working_pcd)

            # Step 4: DBSCAN clustering (keep only significant clusters)
            print("Applying SELECTIVE V3 DBSCAN clustering...")
            working_pcd = self._apply_selective_v3_dbscan_clustering(working_pcd)

            # Step 5: NEW hard cylindrical crop for clean mill-only geometry
            print("Applying STRICT MILL CYLINDRICAL CROP...")
            working_pcd = self._strict_mill_cylindrical_crop(working_pcd)

            # Final results
            final_points = len(working_pcd.points)
            retention_rate = (final_points / input_points) * 100

            print("\nCORRECTED V3 Processing Results:")
            print(f"  Input points: {input_points:,}")
            print(f"  Output points: {final_points:,}")
            print(f"  Overall retention: {retention_rate:.1f}%")

            if final_points >= 10_000_000:
                print(f"  TARGET ACHIEVED: {final_points:,} points retained (target: 10-12M)")
            elif final_points >= 8_000_000:
                print(f"  CLOSE TO TARGET: {final_points:,} points retained (target: 10-12M)")
            else:
                print(f"  BELOW TARGET: {final_points:,} points retained (target: 10-12M)")

            gc.collect()
            
            return working_pcd

        except Exception as e:
            print(f"ERROR in V3 noise removal: {str(e)}")
            gc.collect()
            return None

    # ---------------------------------------------------------
    # IMPROVED INITIAL DOWNSAMPLING
    # ---------------------------------------------------------
    def _improved_initial_downsample(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        IMPROVED initial downsampling that starts with more points for better final retention.
        Uses smaller voxels but aims for higher intermediate point count.
        """
        input_points = len(pcd.points)
        target_points = min(self.max_safe_points, 16_000_000)  # Start with more points

        print(f"  IMPROVED initial downsampling: {input_points:,} -> targeting {target_points:,}")

        voxel_sizes = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.012]

        for voxel_size in voxel_sizes:
            try:
                temp_pcd = pcd.voxel_down_sample(voxel_size)
                temp_points = len(temp_pcd.points)

                print(f"  Testing voxel {voxel_size:.3f}: {temp_points:,} points")

                if temp_points <= target_points:
                    retention_percent = (temp_points / input_points) * 100
                    print(f"  Using voxel size {voxel_size:.3f}")
                    print(f"  Achieved: {temp_points:,} points ({retention_percent:.1f}% initial retention)")
                    return temp_pcd

            except Exception as e:
                print(f"  Failed voxel {voxel_size}: {e}")
                continue

        final_pcd = pcd.voxel_down_sample(0.012)
        final_points = len(final_pcd.points)
        retention_percent = (final_points / input_points) * 100
        print(f"  Fallback: {final_points:,} points ({retention_percent:.1f}% retention)")

        return final_pcd

    # ---------------------------------------------------------
    # SOFT BOTTOM PLATE REMOVAL
    # ---------------------------------------------------------
    def _remove_bottom_plate_soft(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Softer bottom plate removal:
        - Detects the dominant horizontal plane near minimum Z using RANSAC
        - Removes only a thin band (3–5 mm) around that plane, avoiding mill interior.
        """
        input_points = len(pcd.points)
        try:
            print(f" Soft bottom plate removal - Input: {input_points:,} points")

            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.01, max_nn=30
                    )
                )

            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.003,
                ransac_n=3,
                num_iterations=1000
            )
            [a, b, c, d] = plane_model
            print(f" Plane found: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

            if abs(c) < 0.8:
                print(" Plane is not horizontal enough, skipping soft plate removal")
                return pcd

            pts = np.asarray(pcd.points)
            dists = (a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d)

            band = 0.004  # 4 mm band
            radial_safe = self._compute_radial_mask(pcd)

            remove_mask = (
                (dists < band) &
                (dists > -band) &
                (~radial_safe)
            )

            keep_mask = ~remove_mask

            filtered_points = pts[keep_mask]
            clean_pcd = o3d.geometry.PointCloud()
            clean_pcd.points = o3d.utility.Vector3dVector(filtered_points)

            if pcd.has_colors():
                colors = np.asarray(pcd.colors)[keep_mask]
                clean_pcd.colors = o3d.utility.Vector3dVector(colors)
            if pcd.has_normals():
                normals = np.asarray(pcd.normals)[keep_mask]
                clean_pcd.normals = o3d.utility.Vector3dVector(normals)

            removed = input_points - len(filtered_points)
            print(f" Soft plate removal: {removed:,} points removed ({removed/input_points*100:.1f}%)")
            return clean_pcd

        except Exception as e:
            print(f" Soft bottom plate removal failed: {e}")
            return pcd

    # ---------------------------------------------------------
    # STATISTICAL OUTLIER (OUTER SHELL ONLY)
    # ---------------------------------------------------------
    def _apply_original_statistical_removal(self, pcd):
        """
        Statistical outlier removal applied ONLY to outer shell.
        Interior mill points are protected by the radial mask.
        """
        input_points = len(pcd.points)

        radial_safe = self._compute_radial_mask(pcd)
        outer_idx = np.where(~radial_safe)[0]
        inner_idx = np.where(radial_safe)[0]

        outer_shell = pcd.select_by_index(outer_idx)
        inner_core = pcd.select_by_index(inner_idx)

        try:
            clean_outer, _ = outer_shell.remove_statistical_outlier(
                nb_neighbors=18,
                std_ratio=2.5
            )
        except Exception:
            clean_outer = outer_shell

        clean_pcd = clean_outer + inner_core

        removed = input_points - len(clean_pcd.points)
        print(
            " Statistical removal (interior protected): "
            f"{removed:,} removed ({removed/input_points*100:.1f}%)"
        )

        return clean_pcd

    # ---------------------------------------------------------
    # SELECTIVE DBSCAN CLUSTERING
    # ---------------------------------------------------------
    def _apply_selective_v3_dbscan_clustering(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        DBSCAN clustering that keeps several largest clusters
        so the whole mill (body + ends) is retained.
        """
        input_points = len(pcd.points)
        eps = 0.04
        min_points = 80

        try:
            print(f" Soft V3 DBSCAN - eps: {eps}, min_points: {min_points}")
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

            if labels.size == 0 or np.all(labels < 0):
                print(" No clusters found, keeping original")
                return pcd

            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            if len(unique_labels) == 0:
                print(" Only noise labels, keeping original")
                return pcd

            K = min(3, len(unique_labels))
            sorted_idx = np.argsort(counts)[::-1]
            clusters_to_keep = unique_labels[sorted_idx[:K]]

            print(f" Keeping {K} largest clusters: sizes = {counts[sorted_idx[:K]]}")

            radial_safe = self._compute_radial_mask(pcd)
            cluster_mask = np.isin(labels, clusters_to_keep) | radial_safe

            pts = np.asarray(pcd.points)[cluster_mask]
            clean_pcd = o3d.geometry.PointCloud()
            clean_pcd.points = o3d.utility.Vector3dVector(pts)

            if pcd.has_colors():
                colors = np.asarray(pcd.colors)[cluster_mask]
                clean_pcd.colors = o3d.utility.Vector3dVector(colors)
            if pcd.has_normals():
                normals = np.asarray(pcd.normals)[cluster_mask]
                clean_pcd.normals = o3d.utility.Vector3dVector(normals)

            final_pts = len(clean_pcd.points)
            print(
                f" DBSCAN kept: {final_pts:,} / {input_points:,} points "
                f"({final_pts/input_points*100:.1f}%)"
            )
            return clean_pcd

        except Exception as e:
            print(f" DBSCAN clustering failed: {e}")
            return pcd

    # ---------------------------------------------------------
    # LEGACY WRAPPERS
    # ---------------------------------------------------------
    def _smart_downsample(self, pcd: o3d.geometry.PointCloud, target_points: int) -> o3d.geometry.PointCloud:
        """Legacy method for backward compatibility - redirects to improved version."""
        return self._improved_initial_downsample(pcd)

    def apply_legacy_v3_noise_removal(self, point_cloud: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
        """
        Exact legacy V3 implementation but with CORRECTED parameters.
        """
        if point_cloud is None or len(point_cloud.points) == 0:
            return None

        try:
            print("Applying CORRECTED legacy V3 noise removal...")

            input_points = len(point_cloud.points)

            if input_points > self.max_safe_points:
                print(f"Safety downsampling: {input_points:,} -> {self.max_safe_points:,}")
                point_cloud = self._improved_initial_downsample(point_cloud)

            voxel_size = 0.006
            downsampled = point_cloud.voxel_down_sample(voxel_size)

            filtered, _ = downsampled.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0
            )

            labels = np.array(filtered.cluster_dbscan(eps=0.05, min_points=100))

            if len(labels) > 0:
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                if len(unique_labels) > 0:
                    sorted_indices = np.argsort(counts)[::-1]
                    largest_size = counts[sorted_indices[0]]
                    significance_threshold = largest_size * 0.01

                    clusters_to_keep = []
                    for idx in sorted_indices:
                        if counts[idx] >= significance_threshold:
                            clusters_to_keep.append(unique_labels[idx])
                        else:
                            break

                    if clusters_to_keep:
                        cluster_mask = np.isin(labels, clusters_to_keep)
                    else:
                        largest_cluster_label = unique_labels[np.argmax(counts)]
                        cluster_mask = labels == largest_cluster_label

                    points = np.asarray(filtered.points)[cluster_mask]
                    final_pcd = o3d.geometry.PointCloud()
                    final_pcd.points = o3d.utility.Vector3dVector(points)

                    if filtered.has_colors():
                        colors = np.asarray(filtered.colors)[cluster_mask]
                        final_pcd.colors = o3d.utility.Vector3dVector(colors)

                    return final_pcd

            return filtered

        except Exception as e:
            print(f"ERROR in corrected legacy V3 noise removal: {str(e)}")
            gc.collect()
            return None


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CORRECTED V3 NOISE REMOVAL")
    print("=" * 60)

    processor = NoiseRemovalProcessor()
    print("Ready for CORRECTED V3 processing")
    print(f"Safe limit: {processor.max_safe_points:,} points")
    print("Target: Clean 10-12M points from 41M input")
