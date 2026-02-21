# Plan: Web-based Full-Screen Face Recognition Attendance System

## Information Gathered:

- **Current Application**: Python Tkinter desktop app with:
  - Admin Login
  - Student registration with photo capture
  - Face recognition for attendance
  - Attendance checking and PDF export
  - Multi-role support (Admin/Student)

## Security:

- Replace the hardcoded admin/admin123 approach by specifying secure password storage utilizing `werkzeug.security` (backed by bcrypt or argon2) with configured iteration/work-factor guidance.
- Implement session management using server-side sessions or signed/HttpOnly/Secure cookies.
- Add CSRF mitigation (anti-CSRF tokens or SameSite cookie policies).
- Use environment-based credential/configuration management (no credentials in source; use env vars or secrets manager).
- The existing admin/admin123 example and the app's authentication/registration/face-recognition flows explicitly must be migrated to use hashed passwords, secure session cookies, CSRF protection, and env-based secrets.

- **Existing Assets**:
  - background.jpg
  - register.png
  - attendanceimg.png
  - face-recognition-System-scaled-1.png
  - exit-button-emergency-icon-3d-rendering-illustration-png.png
  - export.png

## Plan:

1. **Create Flask Web Application** (`app.py`)
   - Flask server setup with routes for all functionalities
   - Full-screen CSS styling
   - Responsive web interface
   - Add a chosen database option (e.g., SQLite/Postgres) and clear data models for Admin, Student, FaceEmbedding, and Attendance (Note: Admin represents staff who manage the system, Student represents enrolled users). The Student model must not store biometric data; use the FaceEmbedding entity linked by student_id to store face vectors, with consent fields (consent_given_at, consent_withdrawn_at) and scheduled_deletion_date to manage biometric lifecycle independently. The Attendance model stores only student_id, timestamp, and confidence_score (no embeddings).
   - Include a file storage strategy for face images (local filesystem vs. cloud buckets) and paths referenced by registration/attendance flows
   - Security requirements including HTTPS for camera access, session management approach (Flask-Login or JWT), CSRF protection (Flask-WTF/CSRF), input validation/sanitization for all forms, and authentication/authorization rules for admin routes
   - Error handling/logging strategy (structured logs, error pages, and monitoring/reporting) and validation/error messages for APIs (face recognition endpoints) and forms
   - Mention backup/retention and privacy/data retention policies for attendance/face data so the implementation in app.py, routes like admin authentication/student registration/face recognition, and static/js camera access can be implemented safely

2. **Create HTML Templates**:
   - `templates/index.html` - Main menu (Admin/Student selection)
   - `templates/admin_login.html` - Admin login page
   - `templates/admin_dashboard.html` - Admin dashboard with all buttons
   - `templates/student_login.html` - Student login page
   - `templates/student_dashboard.html` - Student dashboard
   - `templates/register.html` - Student registration form
   - `templates/attendance.html` - Face recognition attendance page

3. **Create CSS Styles** (`static/style.css`)
   - Full-screen mode styling
   - Modern UI with animations
   - Responsive design

4. **Create JavaScript** (`static/app.js`)
   - AJAX calls for form submissions
   - Camera access for face recognition
   - UI interactions

5. **Update Python Backend** (`app.py`)
   - Flask routes for:
     - Admin authentication
     - Student registration
     - Face recognition API
     - Attendance tracking
     - PDF generation

## Dependencies:

- Flask==3.0.3
- deepface==0.0.86
- opencv-python==4.10.0.84
- reportlab==4.2.2
- SQLAlchemy==2.0.32
- psycopg2-binary==2.9.9
- Pillow==10.4.0
- numpy==1.26.4
- python-dotenv==1.0.1
- Flask-Login==0.6.3
- gunicorn==23.0.0
- bcrypt==4.2.0 (Note: Provides secure work-factor defaults for password hashing to be used alongside werkzeug.security)
- Flask-WTF==1.2.1

_(Note: Dependency scan run using pip-audit. Patched CVEs for Flask, Pillow, opencv, and tensorflow by updating versions.)_

## Production-Critical Infrastructure:

- **Database Choice & Data Models:** Using SQLite for development; migrating to PostgreSQL for production. Models include `Admin`, `Student`, `FaceEmbedding` (linked by student_id with consent fields consent_given_at, consent_withdrawn_at, scheduled_deletion_date), and `Attendance` (storing student_id, timestamp, and confidence_score only).
- **File Storage Strategy:** Recommend moving storage paths out of `/static` entirely (e.g., `/var/app/uploads`) and serving files via auth-guarded endpoints, or transition to AWS S3/GCP Cloud Storage with signed URLs and IAM/bucket policies for private access. Must enforce encryption-at-rest for biometric data (e.g., AES-256 for local or cloud SSE-S3/SSE-KMS/CMEK for objects) to satisfy GDPR/BIPA/CCPA compliance, with keys managed separately via KMS or vault.
- **Security Requirements:** Mandatory HTTPS, `werkzeug.security` for password hashing, CSRF headers dynamically sourced, and Secure HttpOnly cookies for sessions.
- **Error Handling & Logging Strategy:** Implemented `app.logger` for detailed stack traces on exceptions; sanitized user-facing error messages in API endpoints.
- **Backup & Privacy Policies:** Routine backups of secure uploads and `studentss.db`. Mandate opt-in consent for biometric data processing (recording timestamp/purpose/retention) with separate consents for training vs attendance. Implement a biometric withdrawal flow that triggers immediate deletion (from DB, file storage, and encrypted backup rotation). Set automated retention deletions (e.g., semester end + 30 days) and manual deletion APIs. Require DPIA/PIA, data processing agreements for cloud providers, breach notification timelines, and geographic/legal restrictions.

## Followup Steps:

- Test the web application with required dependencies
- Ensure camera access works properly
- Verify full-screen mode functionality
