import os
from dotenv import load_dotenv
load_dotenv() # Load from .env by default
load_dotenv('.env.local') # Override with .env.local if it exists
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
import cv2
import sqlite3
import datetime
import time
import base64
import csv
from deepface import DeepFace
import warnings
warnings.filterwarnings('ignore')
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.units import inch
import numpy as np
import tempfile
import threading
from werkzeug.security import generate_password_hash, check_password_hash
import io

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')
if not app.secret_key:
    raise ValueError("No SECRET_KEY set for Flask application. Please configure it via environment variables.")

import secrets

@app.before_request
def ensure_csrf_token():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(16)

@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=lambda: session.get('csrf_token', ''))

known_embeddings_cache = {}
known_embeddings_cache_lock = threading.Lock()

def get_student_embeddings(roll_number, image_folder):
    with known_embeddings_cache_lock:
        if roll_number in known_embeddings_cache:
            return known_embeddings_cache[roll_number]
    
    embeddings = []
    if os.path.exists(image_folder):
        for img_file in os.listdir(image_folder):
            img_path = os.path.join(image_folder, img_file)
            try:
                reps = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=True)
                if len(reps) > 0:
                    embeddings.append(reps[0]['embedding'])
            except ValueError:
                pass
            except Exception as e:
                app.logger.exception(f"Error computing face embeddings for {img_path}: {e}")
    
    with known_embeddings_cache_lock:
        known_embeddings_cache[roll_number] = embeddings
    return embeddings

def cosine_distance(source_rep, test_rep):
    if isinstance(source_rep, list):
        source_rep = np.array(source_rep)
    if isinstance(test_rep, list):
        test_rep = np.array(test_rep)
    a = np.dot(source_rep, test_rep)
    norm_s = np.linalg.norm(source_rep)
    norm_t = np.linalg.norm(test_rep)
    denom = norm_s * norm_t
    if denom == 0:
        return 1.0
    return 1 - (a / denom)

def setup_database():
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_number INTEGER UNIQUE NOT NULL,
            department TEXT NOT NULL,
            address TEXT NOT NULL,
            image_folder TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number INTEGER NOT NULL,
            login_time TEXT,
            logout_time TEXT,
            FOREIGN KEY (roll_number) REFERENCES students (roll_number)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    c.execute("SELECT * FROM admin WHERE username = 'admin'")
    if not c.fetchone():
        initial_pw = os.environ.get('INITIAL_TEACHER_PASSWORD') or secrets.token_urlsafe(12)
        with open('teacher_password.txt', 'w') as f:
            f.write(initial_pw)
        try:
            os.chmod('teacher_password.txt', 0o600)
        except Exception:
            pass
        hashed_pw = generate_password_hash(initial_pw)
        c.execute("INSERT INTO admin (username, password) VALUES ('admin', ?)", (hashed_pw,))
    conn.commit()
    conn.close()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/admin_login')
def admin_login_page():
    return render_template('admin_login.html')


@app.route('/admin_login', methods=['POST'])
def admin_login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    c.execute("SELECT password FROM admin WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    
    if result and check_password_hash(result[0], password):
        session['admin_logged_in'] = True
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Invalid credentials'})


@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login_page'))
    return render_template('admin_dashboard.html')


@app.route('/student_login')
def student_login_page():
    return render_template('student_login.html')


@app.route('/student_login', methods=['POST'])
def student_login():
    csrf_token_header = request.headers.get('X-CSRFToken')
    if not csrf_token_header or csrf_token_header == 'MISSING_CSRF_TOKEN' or csrf_token_header != session.get('csrf_token'):
        return jsonify({'success': False, 'message': 'CSRF verification failed or missing token'}), 400
        
    data = request.json
    roll_number = data.get('roll_number')
    image_data = data.get('image')
    
    if not isinstance(roll_number, (int, str)) or not str(roll_number).isdigit():
        return jsonify({'success': False, 'message': 'Invalid roll number'})
        
    if not image_data:
        return jsonify({'success': False, 'message': 'Face verification image required'})
    
    roll_number = int(roll_number)
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    c.execute("SELECT name, roll_number, image_folder FROM students WHERE roll_number = ?", (roll_number,))
    student = c.fetchone()
    conn.close()
    
    if not student:
        return jsonify({'success': False, 'message': 'Student not found'})
        
    try:
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        else:
            image_data = image_data.strip()
        img_bytes = base64.b64decode(image_data)
    except Exception as e:
        app.logger.exception("Decode error")
        return jsonify({'success': False, 'message': 'Invalid image format'})
        
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    cap_path = temp_file.name
    temp_file.write(img_bytes)
    temp_file.close()
    
    try:
        reps = DeepFace.represent(img_path=cap_path, model_name="Facenet", enforce_detection=True)
    except ValueError:
        os.remove(cap_path)
        return jsonify({'success': False, 'message': 'No face detected'})
    except Exception as e:
        app.logger.exception("Face match error")
        if os.path.exists(cap_path):
            os.remove(cap_path)
        return jsonify({'success': False, 'message': 'Verification error'})
        
    cap_emb = reps[0]['embedding']
    student_embeddings = get_student_embeddings(roll_number, student[2])
    
    best_dist = float('inf')
    if student_embeddings:
        for emb in student_embeddings:
            dist = cosine_distance(cap_emb, emb)
            if dist < best_dist:
                best_dist = dist
                
    if os.path.exists(cap_path):
        os.remove(cap_path)
        
    if best_dist <= 0.35:
        session['student_logged_in'] = True
        session['student_name'] = student[0]
        session['student_roll'] = student[1]
        return jsonify({'success': True, 'name': student[0]})
    else:
        return jsonify({'success': False, 'message': 'Face verification failed'})


@app.route('/student_dashboard')
def student_dashboard():
    if not session.get('student_logged_in'):
        return redirect(url_for('student_login_page'))
    return render_template('student_dashboard.html', 
                           student_name=session.get('student_name'), 
                           student_roll=session.get('student_roll'))


@app.route('/register')
def register_page():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login_page'))
    return render_template('register.html')


@app.route('/register', methods=['POST'])
def register_student():
    if not session.get('admin_logged_in'):
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    data = request.json
    name = data.get('name')
    roll_number = data.get('roll_number')
    department = data.get('department')
    address = data.get('address')
    images = data.get('images', [])
    
    if not all([name, roll_number, department, address]):
        return jsonify({'success': False, 'message': 'All fields are required'})
    
    if not isinstance(roll_number, (int, str)) or not str(roll_number).isdigit():
        return jsonify({'success': False, 'message': 'Roll number must be an integer'})
    
    roll_number = int(roll_number)
    
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    c.execute("SELECT roll_number FROM students WHERE roll_number = ?", (roll_number,))
    if c.fetchone():
        conn.close()
        return jsonify({'success': False, 'message': 'Roll number already exists'})
    
    image_folder = os.path.join("known_faces", str(roll_number))
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    for i, img_data in enumerate(images[:5]):
        try:
            if ',' in img_data:
                img_data_clean = img_data.split(',', 1)[1]
            else:
                img_data_clean = img_data.strip()
            img_bytes = base64.b64decode(img_data_clean)
        except Exception as e:
            app.logger.exception(f"Error decoding image base64 for {roll_number}: {e}")
            continue
            
        image_path = os.path.join(image_folder, f"{roll_number}_{i}.jpg")
        with open(image_path, 'wb') as f:
            f.write(img_bytes)
    
    c.execute("INSERT INTO students (name, roll_number, department, address, image_folder) VALUES (?, ?, ?, ?, ?)",
              (name, roll_number, department, address, image_folder))
    conn.commit()
    conn.close()
    
    with known_embeddings_cache_lock:
        if roll_number in known_embeddings_cache:
            del known_embeddings_cache[roll_number]
    
    return jsonify({'success': True, 'message': 'Registration successful!'})


@app.route('/attendance')
def attendance_page():
    if not session.get('admin_logged_in') and not session.get('student_logged_in'):
        return redirect(url_for('index'))
    return render_template('attendance.html')


@app.route('/recognize', methods=['POST'])
def recognize_face():
    if not session.get('admin_logged_in'):
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    try:
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        else:
            image_data = image_data.strip()
        img_bytes = base64.b64decode(image_data)
    except Exception as e:
        app.logger.exception(f"Error decoding base64 image: {e}")
        return jsonify({'success': False, 'message': 'Invalid image format'})
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    captured_image_path = temp_file.name
    temp_file.write(img_bytes)
    temp_file.close()
    
    try:
        captured_reps = DeepFace.represent(img_path=captured_image_path, model_name="Facenet", enforce_detection=True)
    except ValueError:
        os.remove(captured_image_path)
        return jsonify({'success': False, 'message': 'No face detected in the given image'})
    except Exception as e:
        app.logger.exception("An error occurred during face recognition")
        if 'conn' in locals():
            try:
                conn.close()
            except Exception:
                pass
        if os.path.exists(captured_image_path):
            os.remove(captured_image_path)
        return jsonify({'success': False, 'message': 'An internal error occurred'})
            
    captured_embedding = captured_reps[0]['embedding']
    
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    c.execute("SELECT name, roll_number, image_folder FROM students")
    students = c.fetchall()
    
    best_match = None
    best_dist = 0.35  # Stricter verification threshold
    
    for student in students:
        name, roll_number, image_folder = student
        
        student_embeddings = get_student_embeddings(roll_number, image_folder)
        
        if not student_embeddings:
            continue
        
        for emb in student_embeddings:
            dist = cosine_distance(captured_embedding, emb)
            if dist < best_dist:
                best_dist = dist
                best_match = (name, roll_number)
    
    if best_match:
        name, roll_number = best_match
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        c.execute("SELECT id, login_time, logout_time FROM attendance WHERE roll_number = ? AND DATE(login_time) = DATE('now')", (roll_number,))
        record = c.fetchone()
        
        if record:
            if record[2] is None:
                c.execute("UPDATE attendance SET logout_time = ? WHERE id = ?", (current_time, record[0]))
                conn.commit()
                conn.close()
                if os.path.exists(captured_image_path):
                    os.remove(captured_image_path)
                return jsonify({'success': True, 'action': 'logout', 'name': name})
            else:
                conn.close()
                if os.path.exists(captured_image_path):
                    os.remove(captured_image_path)
                return jsonify({'success': True, 'action': 'already_done', 'name': name})
        else:
            c.execute("INSERT INTO attendance (roll_number, login_time) VALUES (?, ?)", (roll_number, current_time))
            conn.commit()
            conn.close()
            if os.path.exists(captured_image_path):
                os.remove(captured_image_path)
            return jsonify({'success': True, 'action': 'login', 'name': name})
    else:
        conn.close()
        if os.path.exists(captured_image_path):
            os.remove(captured_image_path)
        return jsonify({'success': False, 'message': 'No match found'})


@app.route('/student_recognize', methods=['POST'])
def student_recognize():
    if not session.get('student_logged_in'):
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    data = request.json
    image_data = data.get('image')
    roll_number = session.get('student_roll')
    
    if not roll_number:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    if not image_data:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    try:
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        else:
            image_data = image_data.strip()
        img_bytes = base64.b64decode(image_data)
    except Exception as e:
        app.logger.exception(f"Error decoding base64 image: {e}")
        return jsonify({'success': False, 'message': 'Invalid image format'})
        
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    captured_image_path = temp_file.name
    temp_file.write(img_bytes)
    temp_file.close()
    
    try:
        captured_reps = DeepFace.represent(img_path=captured_image_path, model_name="Facenet", enforce_detection=True)
    except ValueError:
        os.remove(captured_image_path)
        return jsonify({'success': False, 'message': 'No face detected in the image'})
    except Exception as e:
        app.logger.exception("An error occurred during student face recognition")
        if 'conn' in locals():
            try:
                conn.close()
            except Exception:
                pass
        if os.path.exists(captured_image_path):
            os.remove(captured_image_path)
        return jsonify({'success': False, 'message': 'An internal error occurred'})
            
    captured_embedding = captured_reps[0]['embedding']
    
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    c.execute("SELECT image_folder FROM students WHERE roll_number = ?", (roll_number,))
    result = c.fetchone()
    
    if not result:
        conn.close()
        if os.path.exists(captured_image_path):
            os.remove(captured_image_path)
        return jsonify({'success': False, 'message': 'Student not found'})
    
    image_folder = result[0]
    student_embeddings = get_student_embeddings(roll_number, image_folder)
    
    best_dist = float('inf')
    if student_embeddings:
        for emb in student_embeddings:
            dist = cosine_distance(captured_embedding, emb)
            if dist < best_dist:
                best_dist = dist
    
    if os.path.exists(captured_image_path):
        os.remove(captured_image_path)
    
    if best_dist <= 0.35:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        c.execute("SELECT id, logout_time FROM attendance WHERE roll_number = ? AND DATE(login_time) = DATE('now')", (roll_number,))
        record = c.fetchone()
        
        if record and record[1] is None:
            c.execute("UPDATE attendance SET logout_time = ? WHERE id = ?", (current_time, record[0]))
            conn.commit()
            conn.close()
            return jsonify({'success': True, 'action': 'logout'})
        elif record:
            conn.close()
            return jsonify({'success': True, 'action': 'already_done'})
        else:
            c.execute("INSERT INTO attendance (roll_number, login_time) VALUES (?, ?)", (roll_number, current_time))
            conn.commit()
            conn.close()
            return jsonify({'success': True, 'action': 'login'})
    else:
        conn.close()
        return jsonify({'success': False, 'message': 'Face not recognized'})


@app.route('/check_attendance', methods=['POST'])
def check_attendance():
    if not session.get('admin_logged_in'):
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    data = request.json
    roll_number = data.get('roll_number')
    
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    
    if not roll_number:
        # Fetch all logs
        c.execute("""
            SELECT s.name, a.roll_number, a.login_time, a.logout_time, a.id
            FROM attendance a 
            JOIN students s ON a.roll_number = s.roll_number 
            ORDER BY a.login_time DESC
        """)
        records = c.fetchall()
        conn.close()
        
        return jsonify({
            'success': True,
            'name': 'All Students',
            'is_all': True,
            'records': [{'name': r[0], 'roll': r[1], 'login': r[2], 'logout': r[3], 'id': r[4]} for r in records]
        })
    
    if not str(roll_number).isdigit():
        conn.close()
        return jsonify({'success': False, 'message': 'Invalid roll number'})
    
    roll_number = int(roll_number)
    c.execute("SELECT login_time, logout_time, id FROM attendance WHERE roll_number = ? ORDER BY login_time DESC", (roll_number,))
    records = c.fetchall()
    
    c.execute("SELECT name FROM students WHERE roll_number = ?", (roll_number,))
    student = c.fetchone()
    conn.close()
    
    if student:
        return jsonify({
            'success': True,
            'name': student[0],
            'is_all': False,
            'records': [{'login': r[0], 'logout': r[1], 'id': r[2]} for r in records]
        })
    return jsonify({'success': False, 'message': 'Student not found'})


@app.route('/admin_manual_logout', methods=['POST'])
def admin_manual_logout():
    if not session.get('admin_logged_in'):
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    data = request.json
    attendance_id = data.get('attendance_id')
    
    if not attendance_id:
        return jsonify({'success': False, 'message': 'Missing record ID'})
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        conn = sqlite3.connect('studentss.db')
        c = conn.cursor()
        c.execute("UPDATE attendance SET logout_time = ? WHERE id = ? AND logout_time IS NULL", (current_time, attendance_id))
        conn.commit()
        updated = c.rowcount > 0
        conn.close()
        
        if updated:
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Record already has logout time or not found'})
    except Exception as e:
        app.logger.exception("Manual logout error")
        return jsonify({'success': False, 'message': 'Internal error'})


@app.route('/get_student/<int:roll_number>')
def get_student(roll_number):
    if not session.get('admin_logged_in'):
        return jsonify({'success': False})
    
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    c.execute("SELECT name, department, address FROM students WHERE roll_number = ?", (roll_number,))
    student = c.fetchone()
    conn.close()
    
    if student:
        return jsonify({'success': True, 'student': {'name': student[0], 'department': student[1], 'address': student[2]}})
    return jsonify({'success': False})


@app.route('/edit_student', methods=['POST'])
def edit_student():
    if not session.get('admin_logged_in'):
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    data = request.json
    roll_number = data.get('roll_number')
    name = data.get('name')
    department = data.get('department')
    address = data.get('address')
    
    if not isinstance(roll_number, (int, str)) or not str(roll_number).isdigit():
        return jsonify({'success': False, 'message': 'Invalid roll number'}), 400
    
    if not name or not isinstance(name, str) or not name.strip():
        return jsonify({'success': False, 'message': 'Invalid name'}), 400
        
    if not department or not isinstance(department, str) or not department.strip():
        return jsonify({'success': False, 'message': 'Invalid department'}), 400
        
    if not address or not isinstance(address, str) or not address.strip():
        return jsonify({'success': False, 'message': 'Invalid address'}), 400
    
    try:
        conn = sqlite3.connect('studentss.db')
        c = conn.cursor()
        c.execute("UPDATE students SET name = ?, department = ?, address = ? WHERE roll_number = ?",
                  (name.strip(), department.strip(), address.strip(), int(roll_number)))
        
        if c.rowcount == 0:
            conn.close()
            return jsonify({'success': False, 'message': 'Student not found'}), 404
            
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        app.logger.exception("Error updating student")
        return jsonify({'success': False, 'message': 'An internal error occurred'})


@app.route('/list_students')
def list_students():
    if not session.get('admin_logged_in'):
        return jsonify({'success': False})
    
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    c.execute("SELECT name, roll_number, department, address FROM students ORDER BY name ASC")
    students = c.fetchall()
    conn.close()
    
    return jsonify({
        'success': True,
        'students': [{'name': s[0], 'roll': s[1], 'dept': s[2], 'address': s[3]} for s in students]
    })


@app.route('/delete_student', methods=['POST'])
def delete_student():
    if not session.get('admin_logged_in'):
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    data = request.json
    roll_number = data.get('roll_number')
    
    if not roll_number:
        return jsonify({'success': False, 'message': 'Roll number required'})
    
    try:
        conn = sqlite3.connect('studentss.db')
        c = conn.cursor()
        
        # 1. Delete attendance records first (foreign key)
        c.execute("DELETE FROM attendance WHERE roll_number = ?", (roll_number,))
        
        # 2. Get image folder path before deleting student record
        c.execute("SELECT image_folder FROM students WHERE roll_number = ?", (roll_number,))
        res = c.fetchone()
        
        if res:
            image_folder = res[0]
            # 3. Delete student record
            c.execute("DELETE FROM students WHERE roll_number = ?", (roll_number,))
            conn.commit()
            
            # 4. Remove image files and folder
            if os.path.exists(image_folder):
                import shutil
                shutil.rmtree(image_folder)
                
            # 5. Clear cache
            with known_embeddings_cache_lock:
                if int(roll_number) in known_embeddings_cache:
                    del known_embeddings_cache[int(roll_number)]
                    
            conn.close()
            return jsonify({'success': True})
        
        conn.close()
        return jsonify({'success': False, 'message': 'Student not found'})
        
    except Exception as e:
        app.logger.exception("Delete student error")
        return jsonify({'success': False, 'message': 'Internal error occurred'})


from flask import make_response

@app.route('/export_report')
def export_report():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login_page'))
    
    roll_number = request.args.get('roll_number')
    export_format = request.args.get('format', 'pdf')
    
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    
    query = """
        SELECT s.name, s.department, s.roll_number, a.login_time, a.logout_time 
        FROM attendance a 
        JOIN students s ON a.roll_number = s.roll_number
        WHERE 1=1
    """
    params = []
    
    if roll_number:
        query += " AND s.roll_number = ?"
        params.append(roll_number)
    

        
    query += " ORDER BY s.name ASC, a.login_time DESC"
    
    c.execute(query, params)
    all_records = c.fetchall()
    conn.close()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if export_format == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Student Name', 'Department', 'Roll Number', 'Login Time', 'Logout Time', 'Duration'])
        
        for name, dept, roll, login, logout in all_records:
            duration_str = "N/A"
            if login and logout:
                login_dt = datetime.datetime.strptime(login, "%Y-%m-%d %H:%M:%S")
                logout_dt = datetime.datetime.strptime(logout, "%Y-%m-%d %H:%M:%S")
                duration = logout_dt - login_dt
                duration_str = str(duration).split('.')[0]
            
            writer.writerow([name, dept, roll, login, logout or 'Active', duration_str])
            
        output.seek(0)
        response = make_response(send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'Attendance_Report_{timestamp}.csv'
        ))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    # PDF Generation
    temp_pdf_fd, pdf_file = tempfile.mkstemp(suffix='.pdf')
    os.close(temp_pdf_fd)
    
    document = SimpleDocTemplate(pdf_file, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=26,
        spaceAfter=15,
        alignment=1,
        textColor=colors.HexColor("#1e293b")
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        alignment=1,
        spaceAfter=30,
        textColor=colors.HexColor("#64748b")
    )
    
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor("#2563eb"),
        borderPadding=5
    )
    
    # Header
    elements.append(Paragraph("Biometric Attendance Report", title_style))
    
    filter_desc = "Complete System Logs"
    if roll_number:
        filter_desc = f"Report for Roll #{roll_number}"

        
    elements.append(Paragraph(filter_desc, subtitle_style))
    elements.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", subtitle_style))
    elements.append(Spacer(1, 10))
    
    # Group by student
    report_data = {}
    for name, dept, roll, login, logout in all_records:
        key = (name, dept, roll)
        if key not in report_data:
            report_data[key] = []
        report_data[key].append((login, logout))
    
    if not report_data:
        elements.append(Paragraph("No records matching the selected criteria.", styles['Normal']))
    else:
        for (name, dept, roll), actions in report_data.items():
            elements.append(Paragraph(f"{name} (Roll: {roll}) - {dept}", header_style))
            
            table_data = [["Date", "Time In", "Time Out", "Duration"]]
            
            total_seconds = 0
            sessions_count = 0
            
            for login_str, logout_str in actions:
                login_dt = datetime.datetime.strptime(login_str, "%Y-%m-%d %H:%M:%S")
                date_str = login_dt.strftime("%b %d, %Y (%a)")
                login_time = login_dt.strftime("%I:%M %p")
                
                if logout_str:
                    logout_dt = datetime.datetime.strptime(logout_str, "%Y-%m-%d %H:%M:%S")
                    logout_time = logout_dt.strftime("%I:%M %p")
                    duration = logout_dt - login_dt
                    diff_sec = duration.total_seconds()
                    total_seconds += diff_sec
                    sessions_count += 1
                    
                    hours = int(diff_sec // 3600)
                    minutes = int((diff_sec % 3600) // 60)
                    duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                else:
                    logout_time = "Active"
                    duration_str = "--"
                
                table_data.append([date_str, login_time, logout_time, duration_str])
            
            # Create Table
            t = Table(table_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.2*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#475569")),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('TOPPADDING', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            elements.append(t)
            
            # Summary for student
            if sessions_count > 0:
                avg_sec = total_seconds / sessions_count
                avg_h = int(avg_sec // 3600)
                avg_m = int((avg_sec % 3600) // 60)
                summary_text = f"<b>Sessions:</b> {sessions_count} | <b>Average Duration:</b> {avg_h}h {avg_m}m"
                elements.append(Spacer(1, 5))
                elements.append(Paragraph(summary_text, ParagraphStyle('Summary', parent=styles['Normal'], fontSize=8, alignment=2)))
            
            elements.append(Spacer(1, 20))
            
    document.build(elements)
    
    with open(pdf_file, 'rb') as f:
        pdf_data = f.read()
        
    if os.path.exists(pdf_file):
        try: os.remove(pdf_file)
        except: pass
            
    response = make_response(send_file(
        io.BytesIO(pdf_data),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'Attendance_Report_{timestamp}.pdf'
    ))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response



@app.route('/student_check_attendance', methods=['POST'])
def student_check_attendance():
    if not session.get('student_logged_in'):
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    roll_number = session.get('student_roll')
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    
    c.execute("SELECT login_time, logout_time FROM attendance WHERE roll_number = ? ORDER BY login_time DESC", (roll_number,))
    records = c.fetchall()
    conn.close()
    
    return jsonify({
        'success': True,
        'records': [{'login': r[0], 'logout': r[1]} for r in records]
    })


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/student_logout')
def student_logout():
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    setup_database()
    debug_mode = os.environ.get('APP_DEBUG', 'False').lower() in ['true', '1', 't']
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    
    if host == '0.0.0.0' and debug_mode:
        debug_mode = False
        app.logger.warning("Debug mode disabled because host is 0.0.0.0")
        
    # Note: For production, use a WSGI server like Gunicorn:
    # gunicorn -w 4 -b 0.0.0.0:5000 app:app
    app.run(debug=debug_mode, host=host, port=5000)
