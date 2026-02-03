# Ball Mill Analysis Web Application

Complete pipeline for analyzing ball mill point cloud scans with E57 support, noise removal, alignment, and ML-powered volume prediction.

---

## Overview

This system processes 3D point cloud scans of ball mills through a multi-stage pipeline:

1. **E57 → PLY Conversion** - Converts E57 format to PLY (if needed)
2. **Noise Removal** - Removes outliers and external structures
3. **Alignment** - Aligns scan to reference mill using ICP
4. **Analysis** - Calculates geometry and generates ML predictions
5. **Report Generation** - Creates visual reports with downloadable results

---

## Quick Start

### 1. Project Structure

```
ball_mill_app/
├── app.py                     # Flask web server
├── e57_converter.py           # E57 to PLY converter
├── noise_removal.py           # Noise removal processor
├── align.py                   # Point cloud alignment
├── integ.py                   # Analysis and ML prediction
├── requirements.txt           # Python dependencies
├── templates/
│   ├── login.html            # Login page
│   ├── signup.html           # Sign up page
│   └── dashboard.html        # Main dashboard
├── reference/
│   └── Mill body - vale verde - v1 (2).PLY  # Reference scan (REQUIRED)
├── uploads/                   # Auto-created: uploaded files
└── outputs/                   # Auto-created: processing results
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Reference File

**CRITICAL**: Place your reference mill scan here:
```
reference/Mill body - vale verde - v1 (2).PLY
```

Link to Reference file: 
``` 
https://drive.google.com/drive/folders/1mHlriZLPhHMk7osbKpSgXgF_dPbC5kWU?usp=sharing
```

This file is **required** for the alignment stage.

### 4. Run the Application

```bash
python app.py
```

Access at: `http://localhost:5000`

**Default Login:**
- Email: `demo@example.com`
- Password: `demo123`

---

## Processing Pipeline

### Stage 0: E57 Conversion (Optional)
- **Input**: `.e57` file
- **Output**: `outputs/<job_id>/<filename>_converted.ply`
- **Process**: Reads all scans, combines points, preserves colors

### Stage 1: Noise Removal
- **Input**: Raw PLY file (or converted from E57)
- **Output**: `outputs/<job_id>/<filename>_noise_removed.ply`
- **Process**:
  - Voxel downsampling (targets 10-12M points)
  - Statistical outlier removal (protects interior)
  - DBSCAN clustering (keeps significant clusters)
  - Cylindrical crop (removes external structures)

### Stage 2: Alignment
- **Input**: Noise-removed PLY
- **Output**: `outputs/<job_id>/<filename>_aligned.ply`
- **Process**:
  - PCA-based axis alignment
  - RANSAC global registration (4M iterations)
  - Multi-start ICP (tests 36 rotations)
  - Final refinement (2000 ICP iterations)
  - Centers at origin with consistent orientation
- **Progress**: 40% → 70%

### Stage 3: Analysis & Report
- **Input**: Aligned PLY
- **Outputs**:
  - `outputs/<job_id>/<filename>_analysis.png` (visual report)
  - `outputs/<job_id>/results.json` (raw data)
- **Process**:
  - Geometry extraction (Trial 9: diameter, length)
  - Free height detection (Trial 4: adaptive bottom detection)
  - ML prediction (Gaussian Process Regression on 11 historical scans)
  - 4-panel visualization generation
- **Progress**: 70% → 100%

---

## Output Files

After processing, you'll find these files in `outputs/<job_id>/`:

```
<job_id>/
├── <filename>_converted.ply      # E57 converted (if E57 uploaded)
├── <filename>_noise_removed.ply  # After noise cleaning
├── <filename>_aligned.ply        # After alignment
├── <filename>_analysis.png       # Visual report (4 panels)
└── results.json                  # Analysis data (JSON)
```

---

## Results Provided

### Calculated Parameters:
1. **Effective Diameter (mm)** - Inner diameter after liner correction
2. **Effective Length (mm)** - Length between head planes
3. **Free Height Over Load (mm)** - Distance from charge surface to top
4. **Total Internal Volume (m³)** - Complete mill internal volume
5. **Total Volume Occupied (m³)** - Volume filled by charge
6. **Internal Cylinder Volume (m³)** - Cylindrical body volume
7. **Cylinder Occupied Volume (m³)** - Charge in cylindrical section
8. **Total Mill Filling (%)** - Overall fill percentage
9. **Cylinder Filling (%)** - Cylindrical section fill percentage

### Visual Report Contains:
- **Cross-section (YZ Plane)** - Shows shell fit and charge surface
- **Side View (XZ)** - Shows length and head planes
- **Results Table** - All 9 calculated parameters
- **Fill Schematic** - Cross-sectional view with charge visualization

---


## Technical Details

### Noise Removal Algorithm (V3)
- **Memory-safe**: Handles up to 15M points safely on 32GB RAM
- **Selective**: Protects mill interior while removing external noise
- **Adaptive**: Uses PCA for cylindrical crop
- **Target**: 10-12M clean points from larger raw scans

### Alignment Algorithm
- **Multi-start ICP**: Tests 36 rotational starting positions
- **Global + Local**: RANSAC for coarse, ICP for fine alignment
- **Robust**: Works with partial scans and varying point densities
- **Consistent**: Enforces opening at negative Z

### ML Prediction Model
- **Type**: Gaussian Process Regression
- **Training**: 11 historical mill scans
- **Features**: Diameter, Length, Free Height
- **Outputs**: Corrects geometric volumes based on real measurements

---

##  Troubleshooting

### "Reference file not found"
- Ensure `reference/Mill body - vale verde - v1 (2).PLY` exists
- Check exact filename (case-sensitive, including spaces)

### "E57 conversion failed"
- Verify `pye57` is installed: `pip install pye57`
- Check E57 file is not corrupted

### "Processing failed during alignment"
- Ensure noise-removed file has sufficient points (>100k)
- Check reference file is valid PLY format

### Memory errors
- Adjust `max_safe_points` in `noise_removal.py` line 24
- Reduce to 10M or 8M for systems with <32GB RAM

### Slow processing
- Normal for large files (743MB → ~20-30 minutes total)
- E57 conversion: 1-3 minutes
- Noise removal: 5-10 minutes
- Alignment: 10-15 minutes
- Analysis: 2-3 minutes

---

## File Format Support

### Input Formats:
- **PLY**: Direct processing (recommended)
- **E57**: Auto-converts to PLY, then processes

### Output Formats:
- **PLY**: Intermediate results (noise-removed, aligned)
- **PNG**: Visual report (4-panel analysis)
- **JSON**: Raw numerical results

---

