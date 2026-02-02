#!/usr/bin/env python3


import time
from pathlib import Path
from typing import Optional, Tuple, List, Set

import numpy as np
import open3d as o3d
import pye57
import tkinter as tk
from tkinter import messagebox, filedialog



# ----------------------- E57 CHECK -----------------------

def check_pye57() -> bool:
    try:
        import pye57  # noqa
        return True
    except ImportError:
        return False


# ----------------------- CONVERTER -----------------------

class E57Converter:
    def __init__(self):
        self.processed_files: Set[str] = set()
        self.pye57_available = check_pye57()

        if not self.pye57_available:
            messagebox.showerror(
                "Missing Dependency",
                "pye57 is not installed.\n\nInstall using:\n\npip install pye57"
            )

    def convert(self, e57_path: Path) -> Optional[Path]:
        import pye57

        key = str(e57_path.resolve())
        out_dir = Path(r"output\converted_scans")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{e57_path.stem}_converted.ply"

        if key in self.processed_files and output_path.exists():
            print(f"[SKIP] {e57_path.name}")
            return output_path

        if output_path.exists():
            print(f"[INFO] Using existing: {output_path.name}")
            self.processed_files.add(key)
            return output_path

        print(f"[INFO] Converting {e57_path.name}")
        start = time.time()

        try:
            with pye57.E57(str(e57_path), mode="r") as e57:
                points_list, colors_list = self._read_all_scans(e57)

                if not points_list:
                    print("[ERROR] No points found")
                    return None

                points = np.vstack(points_list)
                colors = np.vstack(colors_list) if colors_list else None

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                if colors is not None:
                    if colors.max() > 1.0:
                        colors = colors / 255.0
                    pcd.colors = o3d.utility.Vector3dVector(colors)

                o3d.io.write_point_cloud(str(output_path), pcd)

                elapsed = time.time() - start
                print(f"[OK] {output_path.name} | {len(points):,} points | {elapsed:.2f}s")

                self.processed_files.add(key)
                return output_path

        except Exception as e:
            print(f"[ERROR] Conversion failed: {e}")
            return None

    # ----------------------- INTERNAL -----------------------

    def _read_all_scans(self, e57):
        points_list = []
        colors_list = []

        scan_count = self._scan_count(e57)
        print(f"[INFO] Scans detected: {scan_count}")

        for i in range(scan_count):
            data = self._read_scan(e57, i)
            if data:
                pts, cols = data
                if len(pts) > 0:
                    points_list.append(pts)
                    if cols is not None:
                        colors_list.append(cols)
                    print(f"  Scan {i + 1}: {len(pts):,} points")

        return points_list, colors_list

    def _scan_count(self, e57) -> int:
        for attr in ("data3d_count", "scan_count"):
            if hasattr(e57, attr):
                val = getattr(e57, attr)
                return val() if callable(val) else val
        return 1

    def _read_scan(self, e57, idx):
        try:
            scan = e57.read_scan(idx, ignore_missing_fields=True)
            return self._extract(scan)
        except Exception:
            return None

    def _extract(self, scan):
        try:
            if not isinstance(scan, dict):
                return None

            if not {"cartesianX", "cartesianY", "cartesianZ"} <= scan.keys():
                return None

            points = np.column_stack([
                scan["cartesianX"],
                scan["cartesianY"],
                scan["cartesianZ"]
            ])

            colors = None
            if {"colorRed", "colorGreen", "colorBlue"} <= scan.keys():
                colors = np.column_stack([
                    scan["colorRed"],
                    scan["colorGreen"],
                    scan["colorBlue"]
                ])

            mask = np.isfinite(points).all(axis=1)
            points = points[mask]
            if colors is not None:
                colors = colors[mask]

            return points, colors

        except Exception:
            return None


# ----------------------- GUI FLOW -----------------------

def select_and_convert():
    root = tk.Tk()
    root.withdraw()

    choice = messagebox.askyesno(
        "E57 Converter",
        "Select input type:\n\nYES → Folder\nNO → Single E57 file"
    )

    converter = E57Converter()
    if not converter.pye57_available:
        return

    if choice:
        folder = filedialog.askdirectory(title="Select folder containing E57 files")
        if not folder:
            return

        folder = Path(folder)
        e57_files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() == ".e57"
        ]

        if not e57_files:
            messagebox.showinfo("No Files", "No .e57 files found in the selected folder.")
            return

        for f in e57_files:
            converter.convert(f)

        messagebox.showinfo(
            "Done",
            f"Converted {len(e57_files)} file(s).\n\nCheck the same folder for PLY files."
        )

    else:
        file = filedialog.askopenfilename(
            title="Select E57 file",
            filetypes=[
                ("E57 files (*.e57)", "*.e57"),
                ("E57 files (*.E57)", "*.E57"),
                ("All files", "*.*")
            ]
        )

        if not file:
            return

        file_path = Path(file)

        if file_path.suffix.lower() != ".e57":
            messagebox.showerror(
                "Invalid File",
                "Selected file is not a valid .e57 file."
            )
            return


        result = converter.convert(file_path)

        if result:
            messagebox.showinfo(
                "Conversion Complete",
                f"Saved:\n{result}"
            )
        else:
            messagebox.showerror(
                "Error",
                "Conversion failed.\nCheck console for details."
            )


# ----------------------- ENTRY POINT -----------------------

if __name__ == "__main__":
    select_and_convert()
