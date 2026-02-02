#!/usr/bin/env python3
"""
Robust Point Cloud Alignment Script (Multi-Start ICP)
Aligns cylindrical scans (mills/kilns) by trying multiple rotational 
starting positions and selecting the best ICP result, then centers
the final aligned scan at the origin, with consistent opening orientation.
"""

import open3d as o3d
import numpy as np
import copy
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from scipy.optimize import least_squares


def find_cylinder_center_yz(points):
    """
    Find the center of the cylindrical shell in the YZ plane.
    Uses RANSAC-like approach to fit circle to outer shell points.
    """
    x = points[:, 0]
    x_mid = np.median(x)
    x_range = np.ptp(x)

    # Use middle 30% of cylinder
    mask = np.abs(x - x_mid) < (0.15 * x_range)
    middle_pts = points[mask]

    if len(middle_pts) < 1000:
        middle_pts = points

    yz_pts = middle_pts[:, 1:]
    r_all = np.sqrt(yz_pts[:, 0] ** 2 + yz_pts[:, 1] ** 2)

    r_threshold = np.percentile(r_all, 95.0)
    shell_mask = r_all >= r_threshold
    shell_pts = yz_pts[shell_mask]

    if len(shell_pts) < 100:
        return np.median(yz_pts[:, 0]), np.median(yz_pts[:, 1])

    def residuals(params):
        cy, cz, radius = params
        return np.sqrt((shell_pts[:, 0] - cy) ** 2 + (shell_pts[:, 1] - cz) ** 2) - radius

    cy_init = np.median(shell_pts[:, 0])
    cz_init = np.median(shell_pts[:, 1])
    r_init = np.median(
        np.sqrt((shell_pts[:, 0] - cy_init) ** 2 + (shell_pts[:, 1] - cz_init) ** 2)
    )

    result = least_squares(
        residuals, [cy_init, cz_init, r_init],
        loss="soft_l1", f_scale=50.0
    )

    cy, cz, _ = result.x
    return cy, cz


def force_center_cylinder_at_origin(pcd):
    """
    Final step: center the cylinder at (0,0,0) using centroid only.
    Shell center in YZ is kept for diagnostics, not used to translate.
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd

    c = pts.mean(axis=0)
    cy_shell, cz_shell = find_cylinder_center_yz(pts)

    print(f"      :: Current centroid: [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]")
    print(f"      :: Shell center YZ (diagnostic): [{cy_shell:.3f}, {cz_shell:.3f}]")

    print(f"      :: Translating by centroid: [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}] -> move centroid to origin")
    pcd.translate(-c)

    pts2 = np.asarray(pcd.points)
    c2 = pts2.mean(axis=0)
    cy2, cz2 = find_cylinder_center_yz(pts2)
    print(f"      :: New centroid: [{c2[0]:.3f}, {c2[1]:.3f}, {c2[2]:.3f}]")
    print(f"      :: New shell YZ (diagnostic): [{cy2:.3f}, {cz2:.3f}]")

    return pcd


def ensure_opening_at_negative_z(pcd):
    """
    Enforce consistent orientation: the end with larger mean radius
    (more 'open') must be at negative Z. If not, flip 180° around X.
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd

    z_vals = pts[:, 2]
    z_min = np.percentile(z_vals, 10)
    z_max = np.percentile(z_vals, 90)

    bottom_pts = pts[z_vals < z_min]
    top_pts = pts[z_vals > z_max]

    if bottom_pts.size == 0 or top_pts.size == 0:
        print("      :: Not enough points in end slices; skipping opening orientation check")
        return pcd

    bottom_r = np.mean(np.linalg.norm(bottom_pts[:, 1:], axis=1))
    top_r = np.mean(np.linalg.norm(top_pts[:, 1:], axis=1))

    print(f"      :: Mean radius bottom (Z−): {bottom_r:.2f}, top (Z+): {top_r:.2f}")

    if top_r > bottom_r:
        open_end = "top"
        open_z_mean = top_pts[:, 2].mean()
    else:
        open_end = "bottom"
        open_z_mean = bottom_pts[:, 2].mean()

    print(f"      :: More open end: {open_end}, mean Z ≈ {open_z_mean:.2f}")

    # Enforce: opening must be at negative Z
    if open_z_mean > 0:
        print("      :: Opening is on positive Z side -> flipping 180° around X-axis...")
        R_flip = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])
        pcd.rotate(R_flip, center=[0, 0, 0])
    else:
        print("      :: Opening already at negative Z - no flip needed")

    return pcd


def align_cylinder_to_reference_axes(reference_pcd, scan_pcd):
    """
    Aligns scan cylinder to match reference orientation:
    - Cylinder axis along X-axis
    - Circular cross-section in YZ plane
    - Roughly centered at origin (final precise centering done later).
    """
    print("      :: Aligning cylinder to reference axes...")

    ref_pts = np.asarray(reference_pcd.points)
    ref_center = ref_pts.mean(axis=0)

    scan_pts = np.asarray(scan_pcd.points)
    scan_center = scan_pts.mean(axis=0)

    aligned_pcd = copy.deepcopy(scan_pcd)
    aligned_pcd.translate(-scan_center)
    print(f"      :: Initial centering at origin (removed): {scan_center}")

    aligned_pts = np.asarray(aligned_pcd.points)
    cov = np.cov(aligned_pts.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    principal_axis = eigenvectors[:, -1]
    if principal_axis[0] < 0:
        principal_axis = -principal_axis

    target_axis = np.array([1.0, 0.0, 0.0])

    if not np.allclose(principal_axis, target_axis, atol=0.01):
        rotation_axis = np.cross(principal_axis, target_axis)
        rotation_norm = np.linalg.norm(rotation_axis)

        if rotation_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_norm
            dot_prod = np.dot(principal_axis, target_axis)
            angle = np.arccos(np.clip(dot_prod, -1.0, 1.0))

            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
            aligned_pcd.rotate(R, center=[0, 0, 0])

            aligned_pts = np.asarray(aligned_pcd.points)
            drift = aligned_pts.mean(axis=0)
            aligned_pcd.translate(-drift)
            print(f"      :: Aligned to X-axis, corrected drift: {drift}")

    ref_pts_centered = ref_pts - ref_center
    ref_cov = np.cov(ref_pts_centered.T)
    _, ref_eigenvectors = np.linalg.eigh(ref_cov)
    ref_secondary = ref_eigenvectors[:, -2]

    aligned_pts = np.asarray(aligned_pcd.points)
    cov = np.cov(aligned_pts.T)
    _, eigenvectors = np.linalg.eigh(cov)
    scan_secondary = eigenvectors[:, -2]

    ref_secondary_yz = ref_secondary.copy()
    ref_secondary_yz[0] = 0
    ref_norm = np.linalg.norm(ref_secondary_yz)

    scan_secondary_yz = scan_secondary.copy()
    scan_secondary_yz[0] = 0
    scan_norm = np.linalg.norm(scan_secondary_yz)

    if ref_norm > 0.1 and scan_norm > 0.1:
        ref_secondary_yz /= ref_norm
        scan_secondary_yz /= scan_norm

        cos_angle = np.dot(scan_secondary_yz, ref_secondary_yz)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        cross = np.cross(scan_secondary_yz, ref_secondary_yz)
        angle_yz = np.arccos(cos_angle)

        if cross[0] < 0:
            angle_yz = -angle_yz

        R_x = o3d.geometry.get_rotation_matrix_from_axis_angle([angle_yz, 0, 0])
        aligned_pcd.rotate(R_x, center=[0, 0, 0])

        aligned_pts = np.asarray(aligned_pcd.points)
        drift = aligned_pts.mean(axis=0)
        aligned_pcd.translate(-drift)
        print(f"      :: YZ alignment, corrected drift: {drift}")

    aligned_pts = np.asarray(aligned_pcd.points)
    final_center = aligned_pts.mean(axis=0)
    if np.linalg.norm(final_center) > 1.0:
        aligned_pcd.translate(-final_center)
        print(f"      :: Final centering correction: {final_center}")

    return aligned_pcd


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5

    print("      :: RANSAC alignment (4M iterations)...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999)
    )
    return result


def multi_start_icp_alignment(scan, ref):
    print("\n[5/7] Multi-Start Rotational ICP Search...")

    axis = np.array([1.0, 0.0, 0.0])

    icp_voxel = 50.0
    scan_down = scan.voxel_down_sample(icp_voxel)
    ref_down = ref.voxel_down_sample(icp_voxel)

    scan_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=icp_voxel * 3, max_nn=30)
    )
    ref_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=icp_voxel * 3, max_nn=30)
    )

    print("      :: Testing rotations every 10° (36 attempts)...")

    best_result = None
    best_fitness = -1.0
    best_angle = 0
    best_transform = np.identity(4)

    for angle_deg in range(0, 360, 10):
        angle_rad = np.deg2rad(angle_deg)

        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle_rad)
        scan_rotated = copy.deepcopy(scan_down)
        scan_rotated.rotate(R, center=[0, 0, 0])

        threshold = 200.0
        icp_result = o3d.pipelines.registration.registration_icp(
            scan_rotated, ref_down, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )

        if icp_result.fitness > best_fitness:
            best_fitness = icp_result.fitness
            best_angle = angle_deg
            best_result = icp_result
            T_rotation = np.identity(4)
            T_rotation[:3, :3] = R
            best_transform = icp_result.transformation @ T_rotation

    print(f"      :: Best starting angle: {best_angle}° (Fitness: {best_fitness:.4f})")

    scan.transform(best_transform)

    print("      :: Final refinement ICP (2000 iterations)...")
    scan_final = scan.voxel_down_sample(icp_voxel)
    ref_final = ref.voxel_down_sample(icp_voxel)

    scan_final.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=icp_voxel * 3, max_nn=30)
    )
    ref_final.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=icp_voxel * 3, max_nn=30)
    )

    final_icp = o3d.pipelines.registration.registration_icp(
        scan_final, ref_final, 200.0, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )

    scan.transform(final_icp.transformation)

    return scan, final_icp


def align_to_reference(reference_pcd, scan_pcd):
    print("\n" + "=" * 80)
    print("ROBUST ALIGNMENT PIPELINE (MULTI-START ICP)")
    print("=" * 80)

    bbox = scan_pcd.get_axis_aligned_bounding_box()
    max_dim = max(bbox.get_extent())
    scaling_factor = 1.0
    if max_dim < 100.0:
        scaling_factor = 1000.0
        print(f"\n[1/7] Detected small units. Scaling by {scaling_factor}...")
    else:
        print("\n[1/7] Units appear correct. No scaling applied.")

    scan_working = copy.deepcopy(scan_pcd)
    ref_working = copy.deepcopy(reference_pcd)
    if scaling_factor != 1.0:
        scan_working.scale(scaling_factor, center=scan_working.get_center())

    print("[2/7] Initial axis alignment and centering...")
    scan_working = align_cylinder_to_reference_axes(ref_working, scan_working)

    ref_working_centered = copy.deepcopy(ref_working)
    ref_center = np.asarray(ref_working.points).mean(axis=0)
    ref_working_centered.translate(-ref_center)
    ref_working_centered = align_cylinder_to_reference_axes(ref_working, ref_working_centered)

    voxel_size = 150.0
    print(f"[3/7] Computing features (Voxel: {voxel_size}mm)...")
    source_down, source_fpfh = preprocess_point_cloud(scan_working, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(ref_working_centered, voxel_size)

    print("[4/7] Running Global RANSAC...")
    ransac_result = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    scan_working.transform(ransac_result.transformation)
    print(f"      RANSAC Fitness: {ransac_result.fitness:.4f}")

    scan_working, final_icp = multi_start_icp_alignment(scan_working, ref_working_centered)
    print(f"      Final ICP Fitness: {final_icp.fitness:.4f}")

    print("[6/7] Final axis re-alignment...")
    scan_working = align_cylinder_to_reference_axes(ref_working_centered, scan_working)

    print("      :: Enforcing opening at negative Z...")
    scan_working = ensure_opening_at_negative_z(scan_working)

    print("[7/7] Forcing cylinder center to origin (0,0,0)...")
    scan_working = force_center_cylinder_at_origin(scan_working)

    final_pts = np.asarray(scan_working.points)
    cov = np.cov(final_pts.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal_axis = eigenvectors[:, -1]
    if principal_axis[0] < 0:
        principal_axis = -principal_axis
    print(
        f"      Principal axis after final centering: "
        f"[{principal_axis[0]:.3f}, {principal_axis[1]:.3f}, {principal_axis[2]:.3f}]"
    )

    print("\n" + "=" * 80)
    print("ALIGNMENT COMPLETED")
    print("=" * 80 + "\n")

    return scan_working


def load_point_cloud(filepath):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    print(f"Loading: {filepath}")
    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.points:
        raise ValueError(f"Empty point cloud: {filepath}")
    print(f"  Points: {len(pcd.points):,}")
    return pcd


def save_point_cloud(pcd, filepath):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {filepath}")
    o3d.io.write_point_cloud(str(path), pcd)


def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return filepath


def main():
    print("\nPOINT CLOUD ALIGNMENT TOOL")
    filetypes = [("Point Clouds", "*.ply *.pcd *.xyz *.pts"), ("All", "*.*")]

    try:
        reference_path = r"reference\Mill body - vale verde - v1 (2).PLY"

        print("\nSelect SCAN file...")
        scan_path = select_file("Select Scan", filetypes)
        if not scan_path:
            return 1

        ref_pcd = load_point_cloud(reference_path)
        scan_pcd = load_point_cloud(scan_path)

        aligned_pcd = align_to_reference(ref_pcd, scan_pcd)

        out_dir = Path(r"output\aligned_scans")
        out_name = f"{Path(scan_path).stem}_aligned.ply"
        save_point_cloud(aligned_pcd, out_dir / out_name)

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
