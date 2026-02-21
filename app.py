import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
import cv2
import sqlite3
import datetime
import time
import base64
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

app = Flask(__name__)
app.secret_key = 'fras_secret_key_2024'

known_embeddings_cache = {}

def get_student_embeddings(roll_number, image_folder):
    if roll_number in known_embeddings_cache:
        return known_embeddings_cache[roll_number]
    
    embeddings = []
    if os.path.exists(image_folder):
        for img_file in os.listdir(image_folder):
            img_path = os.path.join(image_folder, img_file)
            try:
                reps = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
                if len(reps) > 0:
                    embeddings.append(reps[0]['embedding'])
            except:
                pass
    known_embeddings_cache[roll_number] = embeddings
    return embeddings

def cosine_distance(source_rep, test_rep):
    if isinstance(source_rep, list):
        source_rep = np.array(source_rep)
    if isinstance(test_rep, list):
        test_rep = np.array(test_rep)
    a = np.matmul(np.transpose(source_rep), test_rep)
    b = np.sum(np.multiply(source_rep, source_rep))
    c = np.sum(np.multiply(test_rep, test_rep))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

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
    
    c.execute("DELETE FROM admin WHERE username = 'admin'")
    c.execute("INSERT INTO admin (username, password) VALUES ('admin', 'admin123')")
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
    c.execute("DELETE FROM admin WHERE username = 'admin'")
    c.execute("INSERT INTO admin (username, password) VALUES ('admin', 'admin123')")
    conn.commit()
    c.execute("SELECT * FROM admin WHERE username = ? AND password = ?", (username, password))
    result = c.fetchone()
    conn.close()
    
    if result:
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
    data = request.json
    roll_number = data.get('roll_number')
    
    if not roll_number or not roll_number.isdigit():
        return jsonify({'success': False, 'message': 'Invalid roll number'})
    
    roll_number = int(roll_number)
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    c.execute("SELECT name, roll_number FROM students WHERE roll_number = ?", (roll_number,))
    student = c.fetchone()
    conn.close()
    
    if student:
        session['student_logged_in'] = True
        session['student_name'] = student[0]
        session['student_roll'] = student[1]
        return jsonify({'success': True, 'name': student[0]})
    return jsonify({'success': False, 'message': 'Student not found'})


@app.route('/student_dashboard')
def student_dashboard():
    if not session.get('student_logged_in'):
        return redirect(url_for('student_login_page'))
    return render_template('student_dashboard.html')


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
    
    if not roll_number.isdigit():
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
        img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        image_path = os.path.join(image_folder, f"{roll_number}_{i}.jpg")
        with open(image_path, 'wb') as f:
            f.write(img_bytes)
    
    c.execute("INSERT INTO students (name, roll_number, department, address, image_folder) VALUES (?, ?, ?, ?, ?)",
              (name, roll_number, department, address, image_folder))
    conn.commit()
    conn.close()
    
    if roll_number in known_embeddings_cache:
        del known_embeddings_cache[roll_number]
    
    return jsonify({'success': True, 'message': 'Registration successful!'})


@app.route('/attendance')
def attendance_page():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login_page'))
    return render_template('attendance.html')


@app.route('/recognize', methods=['POST'])
def recognize_face():
    if not session.get('admin_logged_in'):
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    image_data = image_data.split(',')[1]
    img_bytes = base64.b64decode(image_data)
    
    captured_image_path = 'known_faces/temp.jpg'
    with open(captured_image_path, 'wb') as f:
        f.write(img_bytes)
    
    try:
        captured_reps = DeepFace.represent(img_path=captured_image_path, model_name="Facenet", enforce_detection=False)
        if not captured_reps:
            os.remove(captured_image_path)
            return jsonify({'success': False, 'message': 'No face detected in the given image'})
            
        captured_embedding = captured_reps[0]['embedding']
        
        conn = sqlite3.connect('studentss.db')
        c = conn.cursor()
        c.execute("SELECT name, roll_number, image_folder FROM students")
        students = c.fetchall()
        
        valid_matches = []
        
        for student in students:
            name, roll_number, image_folder = student
            
            student_embeddings = get_student_embeddings(roll_number, image_folder)
            
            if not student_embeddings:
                continue
            
            student_matches = 0
            student_distances = []
            
            for emb in student_embeddings:
                dist = cosine_distance(captured_embedding, emb)
                if dist <= 0.40:
                    student_matches += 1
                    student_distances.append(dist)
            
            if student_matches >= 1:
                avg_distance = sum(student_distances) / len(student_distances)
                valid_matches.append((avg_distance, name, roll_number))
        
        if valid_matches:
            # Sort by lowest average distance
            valid_matches.sort(key=lambda x: x[0])
            best_distance, name, roll_number = valid_matches[0]
            
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
            
    except Exception as e:
        if os.path.exists(captured_image_path):
            os.remove(captured_image_path)
        return jsonify({'success': False, 'message': str(e)})


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
    
    image_data = image_data.split(',')[1]
    img_bytes = base64.b64decode(image_data)
    
    captured_image_path = 'known_faces/temp_student.jpg'
    with open(captured_image_path, 'wb') as f:
        f.write(img_bytes)
    
    try:
        captured_reps = DeepFace.represent(img_path=captured_image_path, model_name="Facenet", enforce_detection=False)
        if not captured_reps:
            os.remove(captured_image_path)
            return jsonify({'success': False, 'message': 'No face detected in the image'})
            
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
        
        matches = 0
        if student_embeddings:
            for emb in student_embeddings:
                dist = cosine_distance(captured_embedding, emb)
                if dist <= 0.40:
                    matches += 1
        
        if os.path.exists(captured_image_path):
            os.remove(captured_image_path)
        
        if matches >= 1:
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
            
    except Exception as e:
        if os.path.exists(captured_image_path):
            os.remove(captured_image_path)
        return jsonify({'success': False, 'message': str(e)})


@app.route('/check_attendance', methods=['POST'])
def check_attendance():
    if not session.get('admin_logged_in'):
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    data = request.json
    roll_number = data.get('roll_number')
    
    if not roll_number or not roll_number.isdigit():
        return jsonify({'success': False, 'message': 'Invalid roll number'})
    
    roll_number = int(roll_number)
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    
    c.execute("SELECT login_time, logout_time FROM attendance WHERE roll_number = ? ORDER BY login_time DESC", (roll_number,))
    records = c.fetchall()
    
    c.execute("SELECT name FROM students WHERE roll_number = ?", (roll_number,))
    student = c.fetchone()
    conn.close()
    
    if student:
        return jsonify({
            'success': True,
            'name': student[0],
            'records': [{'login': r[0], 'logout': r[1]} for r in records]
        })
    return jsonify({'success': False, 'message': 'Student not found'})


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
    
    try:
        conn = sqlite3.connect('studentss.db')
        c = conn.cursor()
        c.execute("UPDATE students SET name = ?, department = ?, address = ? WHERE roll_number = ?",
                  (name, department, address, roll_number))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/export_pdf')
def export_pdf():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login_page'))
    
    pdf_file = "attendance_report.pdf"
    document = SimpleDocTemplate(pdf_file, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,
        textColor=colors.darkblue
    )
    
    title = Paragraph("Face Recognition Attendance System Report", title_style)
    elements.append(title)
    
    current_date = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
    date_style = ParagraphStyle('DateStyle', parent=styles['Normal'], fontSize=12, alignment=1, spaceAfter=20)
    date_para = Paragraph(f"Generated on: {current_date}", date_style)
    elements.append(date_para)
    elements.append(Spacer(1, 20))
    
    conn = sqlite3.connect('studentss.db')
    c = conn.cursor()
    c.execute("SELECT name, department, roll_number FROM students")
    students = c.fetchall()
    
    for student_index, student in enumerate(students):
        name, department, roll_number = student
        
        student_style = ParagraphStyle(
            'StudentHeader',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=10,
            textColor=colors.darkgreen
        )
        
        student_header = Paragraph(f"{name} (Roll: {roll_number}) - {department} Department", student_style)
        elements.append(student_header)
        
        c.execute("""
            SELECT login_time, logout_time 
            FROM attendance 
            WHERE roll_number = ? 
            ORDER BY login_time DESC
        """, (roll_number,))
        attendance_records = c.fetchall()
        
        if attendance_records:
            attendance_data = [["Date", "Day", "Login Time", "Logout Time", "Duration"]]
            
            for login_time_str, logout_time_str in attendance_records:
                login_dt = datetime.datetime.strptime(login_time_str, "%Y-%m-%d %H:%M:%S")
                date_str = login_dt.strftime("%b %d, %Y")
                day_str = login_dt.strftime("%A")
                login_display = login_dt.strftime("%I:%M %p")
                
                if logout_time_str:
                    logout_dt = datetime.datetime.strptime(logout_time_str, "%Y-%m-%d %H:%M:%S")
                    logout_display = logout_dt.strftime("%I:%M %p")
                    duration = logout_dt - login_dt
                    duration_hours = duration.total_seconds() / 3600
                    if duration_hours < 1:
                        duration_str = f"{int(duration.total_seconds() / 60)} min"
                    else:
                        hours = int(duration_hours)
                        minutes = int((duration_hours - hours) * 60)
                        duration_str = f"{hours}h {minutes}m"
                else:
                    logout_display = "Not logged out"
                    duration_str = "N/A"
                
                attendance_data.append([date_str, day_str, login_display, logout_display, duration_str])
            
            attendance_table = Table(attendance_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            attendance_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            
            elements.append(attendance_table)
            elements.append(Spacer(1, 20))
        else:
            no_data_style = ParagraphStyle('NoData', parent=styles['Normal'], fontSize=10, spaceAfter=20, leftIndent=20, textColor=colors.red)
            no_data = Paragraph("No attendance records found.", no_data_style)
            elements.append(no_data)
        
        if student_index < len(students) - 1:
            elements.append(Spacer(1, 30))
    
    conn.close()
    document.build(elements)
    
    return send_file(pdf_file, as_attachment=True)


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
    app.run(debug=True, host='0.0.0.0', port=5000)
