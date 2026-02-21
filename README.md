# Biometric Face Recognition Attendance System (FRAS) 📸

A professional, high-end web-based attendance management system using state-of-the-art face recognition technology. This system provides a seamless experience for both administrators and students, featuring a futuristic scanner, comprehensive reporting, and a clean, modular UI.

---

## 🚀 Key Features

### 🔐 Multi-Role Access

- **Administrator Dashboard**: Full control over student records, attendance logs, and system exports.
- **Student Portal**: Personal attendance history tracking and profile management.

### 🤖 Advanced Face Recognition

- **DeepFace Integration**: Uses VGG-Face models for high-accuracy biometric verification.
- **Multi-Angle Enrollment**: Students register with 5 different face samples to ensure robust recognition.
- **Real-time Scanner**: Immersive camera interface with face-positioning guides and live detection overlays.

### 📊 Reporting & Analytics

- **Professional PDF Exports**: Generate high-fidelity attendance reports with session summaries and durations.
- **CSV Data Spreadsheets**: Export log data for external analysis (Excel/Google Sheets).
- **Granular Filtering**: Export reports for the entire system or specific students.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask (Web Framework)
- **AI / Computer Vision**:
  - [DeepFace](https://github.com/serengil/deepface) (Facial Recognition)
  - [OpenCV](https://opencv.org/) (Computer Vision)
  - [TensorFlow](https://www.tensorflow.org/) (ML Backend)
- **Database**: SQLite3 (Lightweight, Relational)
- **Frontend**:
  - Responsive HTML5 & Vanilla CSS3
  - JavaScript (ES6+)
  - [Lucide Icons](https://lucide.dev/) (Modern UI Icons)
- **Reporting**: [ReportLab](https://www.reportlab.com/) (Dynamic PDF Generation)

---

## ⚙️ Installation & Setup

### 1. Prerequisites

- Python 3.9 through 3.12 (TensorFlow compatibility).
- A webcam or integrated laptop camera.

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd Python-Dev-Face-Recognition
```

### 3. Install Dependencies

It is highly recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/Scripts/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configuration

Create a `.env` file in the root directory (refer to `.env.example`):

```env
SECRET_KEY=your_secure_flask_key
INITIAL_ADMIN_PASSWORD=your_admin_password
APP_DEBUG=False
FLASK_HOST=127.0.0.1
```

### 5. Running the Application

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`.

---

## 🛡️ Administrative Setup

### Initial Login

- **Default Username**: `admin`
- **Password**:
  - If you set `INITIAL_ADMIN_PASSWORD` in your `.env`, use that value.
  - Otherwise, check the generated `initial_admin_password.txt` file in the root directory after the first launch.

### Student Enrollment

1. Log in as Admin and navigate to **Register Student**.
2. Enter student details (Roll Number must be unique).
3. Capture exactly **5 face samples** at slightly different angles as prompted.
4. Click **Complete Registration**.

### Marking Attendance

1. Navigate to **Mark Attendance** from the dashboard or home page.
2. Ensure the face is centered within the on-screen guide.
3. The system will automatically detect the student, record the login time, and update the session upon logout recognition.

---

## 📂 Project Structure

- `app.py`: Main Flask application core.
- `templates/`: HTML5 Jinja2 templates.
- `static/`: CSS styling and client-side assets.
- `known_faces/`: Secure storage for encrypted face sample directories.
- `studentss.db`: SQLite database for persistent storage.

---

## 📜 License

This project is intended for educational and developmental purposes. Ensure compliance with local data privacy regulations (GDPR/APPI) when handling biometric data.
