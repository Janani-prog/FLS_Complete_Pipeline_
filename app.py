"""
Ball Mill Analysis Web Application
Flask backend that integrates noise removal, alignment, and analysis
"""

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
from pathlib import Path
from datetime import datetime
import traceback
import threading
import uuid

# Import your existing modules
from noise_removal import NoiseRemovalProcessor
from align import align_to_reference, load_point_cloud as align_load_pc, save_point_cloud
from integ import load_point_cloud, correct_yz_alignment, find_head_planes_trial9, fit_shell_circle_trial9
from integ import find_head_planes_trial4, fit_shell_circle_trial4, compute_free_height_trial4
from integ import predict_single_mill, make_plots, get_theoretical_geometry
from e57_converter import E57Converter, check_pye57

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('outputs')
REFERENCE_PLY = Path(r"reference\Mill body - vale verde - v1 (2).PLY")

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size

# Simple user database 
USERS_DB = {
    'demo@example.com': {
        'password': generate_password_hash('demo123'),
        'name': 'Demo User'
    }
}

# Track processing jobs
processing_jobs = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['ply', 'e57']


@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        user = USERS_DB.get(email)
        if user and check_password_hash(user['password'], password):
            session['user'] = {
                'email': email,
                'name': user['name']
            }
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        
        if email in USERS_DB:
            return jsonify({'success': False, 'error': 'Email already exists'}), 400
        
        USERS_DB[email] = {
            'password': generate_password_hash(password),
            'name': name
        }
        
        session['user'] = {
            'email': email,
            'name': name
        }
        return jsonify({'success': True})
    
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only .ply and .e57 files are allowed'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = app.config['UPLOAD_FOLDER'] / unique_filename
    file.save(filepath)
    
    # Initialize job tracking
    processing_jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'stage': 'Initializing',
        'filename': filename,
        'filepath': str(filepath),
        'user': session['user']['email']
    }
    
    # Start processing in background thread
    thread = threading.Thread(target=process_scan, args=(job_id, filepath, filename))
    thread.daemon = True
    thread.start()
    
    return jsonify({'job_id': job_id, 'filename': filename})


def process_scan(job_id, input_path, original_filename):
    """Process the uploaded scan through all stages"""
    try:
        job = processing_jobs[job_id]
        output_dir = app.config['OUTPUT_FOLDER'] / job_id
        output_dir.mkdir(exist_ok=True)
        
        # Determine file extension
        file_ext = Path(original_filename).suffix.lower()
        
        # Stage 0: E57 to PLY Conversion (if needed)
        if file_ext == '.e57':
            job['stage'] = 'E57 Conversion'
            job['progress'] = 5
            print(f"[{job_id}] E57 to PLY Conversion")
            
            from e57_converter import E57Converter
            converter = E57Converter()
            if not converter.pye57_available:
                raise Exception("pye57 is not installed. Please install it using 'pip install pye57'")
            
            result = converter.convert(input_path)
            if result is None:
                raise Exception("E57 to PLY conversion failed")
            
            input_path = result  # Update input_path for next stages
            job['converted_path'] = str(result)
            job['progress'] = 10
            print(f"E57 converted successfully: {result}")
        
        # Stage 1: Noise Removal
        job['stage'] = 'Noise Removal'
        job['progress'] = 15 if file_ext == '.e57' else 10
        
        print(f"\n{'='*60}\nJob {job_id}: Starting Noise Removal\n{'='*60}")
        
        processor = NoiseRemovalProcessor()
        pcd_raw = align_load_pc(str(input_path))
        pcd_clean = processor.apply_v3_noise_removal(pcd_raw)
        
        if pcd_clean is None:
            raise Exception("Noise removal failed")
        
        noise_removed_path = output_dir / f"{Path(original_filename).stem}_noise_removed.ply"
        save_point_cloud(pcd_clean, str(noise_removed_path))
        job['noise_removed_path'] = str(noise_removed_path)
        job['progress'] = 40 if file_ext == '.e57' else 35
        
        # Stage 2: Alignment
        job['stage'] = 'Alignment'
        job['progress'] = 45 if file_ext == '.e57' else 40
        
        print(f"\n{'='*60}\nJob {job_id}: Starting Alignment\n{'='*60}")
        
        ref_pcd = align_load_pc(str(REFERENCE_PLY))
        scan_pcd = align_load_pc(str(noise_removed_path))
        aligned_pcd = align_to_reference(ref_pcd, scan_pcd)
        
        aligned_path = output_dir / f"{Path(original_filename).stem}_aligned.ply"
        save_point_cloud(aligned_pcd, str(aligned_path))
        job['aligned_path'] = str(aligned_path)
        job['progress'] = 70 if file_ext == '.e57' else 65
        
        # Stage 3: Analysis & Report Generation
        job['stage'] = 'Analysis & Report Generation'
        job['progress'] = 75 if file_ext == '.e57' else 70
        
        print(f"\n{'='*60}\nJob {job_id}: Starting Analysis\n{'='*60}")
        
        points = load_point_cloud(str(aligned_path))
        points = correct_yz_alignment(points)
        
        # Geometry Analysis (Trial 9)
        x_min_t9, x_max_t9, effective_length = find_head_planes_trial9(points)
        cy_t9, cz_t9, radius_t9 = fit_shell_circle_trial9(points, x_min_t9, x_max_t9)
        effective_diameter = 2.0 * radius_t9
        
        job['progress'] = 85 if file_ext == '.e57' else 80
        
        # Free Height Analysis (Trial 4)
        x_min_t4, x_max_t4 = find_head_planes_trial4(points)
        cy_t4, cz_t4, radius_t4 = fit_shell_circle_trial4(points, x_min_t4, x_max_t4)
        free_height = compute_free_height_trial4(points, x_min_t4, x_max_t4, cy_t4, cz_t4, radius_t4)
        
        job['progress'] = 90 if file_ext == '.e57' else 85
        
        # ML Prediction
        ml_results = predict_single_mill(effective_diameter, effective_length, free_height)
        
        # Prepare results
        results = {
            "cy": cy_t9,
            "cz": cz_t9,
            "radius": radius_t9,
            "xmin": x_min_t9,
            "xmax": x_max_t9,
            "zchargesurface": cz_t4 + radius_t4 - free_height,
            
            # Use new key names from updated integ.py
            "effective_diameter_mm": effective_diameter,
            "effective_length_mm": effective_length,
            "free_height_mm": free_height,
            "total_volume_m3": ml_results['total_volume_m3'],
            "charge_total_m3": ml_results['charge_total_m3'],
            "cylinder_volume_m3": ml_results['cylinder_volume_m3'],
            "charge_cylinder_m3": ml_results['charge_cylinder_m3'],
            "total_fill_pct": ml_results['total_fill_pct'],
            "cylinder_fill_pct": ml_results['cylinder_fill_pct']
        }
        
        job['progress'] = 95 if file_ext == '.e57' else 90
        
        # Generate visualization
        report_path = output_dir / f"{Path(original_filename).stem}_analysis.png"
        make_plots(points, results, report_path, original_filename)
        
        # Save results as JSON
        results_json_path = output_dir / "results.json"
        with open(results_json_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {k: float(v) if isinstance(v, (float, int)) else v 
                          for k, v in results.items()}
            json.dump(json_results, f, indent=2)
        
        job['results'] = results
        job['report_path'] = str(report_path)
        job['results_json_path'] = str(results_json_path)
        job['progress'] = 100
        job['status'] = 'completed'
        job['stage'] = 'Complete'
        
        print(f"\n{'='*60}\nJob {job_id}: Processing Complete\n{'='*60}")
        
    except Exception as e:
        print(f"\n{'='*60}\nJob {job_id}: ERROR\n{'='*60}")
        traceback.print_exc()
        job['status'] = 'failed'
        job['error'] = str(e)
        job['stage'] = 'Failed'


@app.route('/status/<job_id>')
def get_status(job_id):
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    job = processing_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    # Only return status info, not full results
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'stage': job['stage'],
        'filename': job['filename'],
        'error': job.get('error')
    })


@app.route('/results/<job_id>')
def get_results(job_id):
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    job = processing_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed'}), 400
    
    return jsonify({
        'status': 'completed',
        'results': job['results'],
        'report_available': True
    })


@app.route('/download/<job_id>/<file_type>')
def download_file(job_id, file_type):
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    job = processing_jobs.get(job_id)
    if not job or job['status'] != 'completed':
        return jsonify({'error': 'Results not available'}), 404
    
    if file_type == 'report':
        return send_file(job['report_path'], as_attachment=True, 
                        download_name=f"report_{job['filename']}.png")
    elif file_type == 'json':
        return send_file(job['results_json_path'], as_attachment=True,
                        download_name=f"results_{job['filename']}.json")
    elif file_type == 'aligned':
        return send_file(job['aligned_path'], as_attachment=True,
                        download_name=f"aligned_{job['filename']}")
    
    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    print("="*60)
    print("BALL MILL ANALYSIS WEB APPLICATION")
    print("="*60)
    print(f"Upload folder: {UPLOAD_FOLDER.absolute()}")
    print(f"Output folder: {OUTPUT_FOLDER.absolute()}")
    print(f"Reference PLY: {REFERENCE_PLY.absolute()}")
    print("="*60)
    print("Starting server on http://localhost:5000")
    print("Default login: demo@example.com / demo123")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
