from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
from datetime import datetime
import numpy as np
from PIL import Image
import io
import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import onnxruntime as ort

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    conn = sqlite3.connect('lunny.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        patient_id TEXT,
        full_name TEXT,
        dob TEXT,
        gender TEXT,
        contact TEXT,
        address TEXT,
        emergency_contact TEXT,
        insurance TEXT,
        smoking_history TEXT,
        occupational_exposure TEXT,
        family_history TEXT,
        previous_conditions TEXT,
        medications TEXT,
        allergies TEXT,
        past_surgeries TEXT,
        scan_type TEXT,
        scan_date TEXT,
        radiologist_notes TEXT,
        ai_output TEXT,
        biopsy_performed TEXT,
        biopsy_result TEXT,
        histology TEXT,
        genetic_tests TEXT,
        cbc TEXT,
        cea_level TEXT,
        other_markers TEXT,
        diagnosis_date TEXT,
        cancer_stage TEXT,
        treatment_type TEXT,
        treatment_start TEXT,
        treatment_response TEXT,
        followup_scans TEXT,
        next_appointment TEXT,
        survivorship_notes TEXT,
        palliative_notes TEXT,
        consent_forms TEXT,
        doctors TEXT,
        referrals TEXT,
        medical_notes TEXT,
        created_at TIMESTAMP,
        passport TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password_hash TEXT,
        hospital_name TEXT,
        admin_id TEXT,
        created_at TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    logger.info("Serving index.html")
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        hospital_name = data.get('hospitalName')
        admin_id = data.get('adminId')

        if not email or not password or not hospital_name or not admin_id:
            return jsonify({'error': 'Missing required fields'}), 400

        password_hash = generate_password_hash(password)

        conn = sqlite3.connect('lunny.db')
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (email, password_hash, hospital_name, admin_id, created_at) VALUES (?, ?, ?, ?, ?)',
                      (email, password_hash, hospital_name, admin_id, datetime.now()))
            conn.commit()
            return jsonify({'message': 'Registration successful'}), 200
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Email already registered'}), 400
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': 'Missing email or password'}), 400

        conn = sqlite3.connect('lunny.db')
        c = conn.cursor()
        c.execute('SELECT password_hash FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()

        if not user or not check_password_hash(user[0], password):
            return jsonify({'error': 'Incorrect credentials'}), 401

        return jsonify({'message': 'Sign In successful'}), 200
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_record', methods=['POST'])
def save_record():
    try:
        data = request.get_json()
        if not data or not data.get('email') or not data.get('patientId') or not data.get('fullName'):
            return jsonify({'error': 'Missing required fields'}), 400

        conn = sqlite3.connect('lunny.db')
        c = conn.cursor()

        c.execute('SELECT id FROM records WHERE patient_id = ? AND email = ?', (data['patientId'], data['email']))
        existing = c.fetchone()

        passport = data.get('passport', '')
        if passport and not passport.startswith('data:image/'):
            passport = ''

        fields = (
            data['email'],
            data['patientId'],
            data['fullName'],
            data.get('dob', ''),
            data.get('gender', ''),
            data.get('contactInfo', ''),
            data.get('address', ''),
            data.get('emergencyContact', ''),
            data.get('insuranceInfo', ''),
            data.get('smokingHistory', ''),
            data.get('occupationalExposure', ''),
            data.get('familyHistory', ''),
            data.get('previousConditions', ''),
            data.get('currentMedications', ''),
            data.get('allergies', ''),
            data.get('pastSurguries', ''),
            data.get('scanType', ''),
            data.get('scanDate', ''),
            data.get('radiologistNotes', ''),
            data.get('aiOutput', ''),
            data.get('biopsyPerformed', ''),
            data.get('biopsyResult', ''),
            data.get('histology', ''),
            data.get('geneticTests', ''),
            data.get('cbc', ''),
            data.get('ceaLevel', ''),
            data.get('otherMarkers', ''),
            data.get('diagnosisDate', ''),
            data.get('cancerStage', ''),
            data.get('treatmentType', ''),
            data.get('treatmentStartDate', ''),
            data.get('responseToTreatment', ''),
            data.get('followUpScans', ''),
            data.get('nextAppointment', ''),
            data.get('survivorshipCare', ''),
            data.get('palliativeNotes', ''),
            data.get('consentForms', ''),
            data.get('doctorInCharge', ''),
            data.get('referrals', ''),
            data.get('medicalNotes', ''),
            datetime.now(),
            passport
        )

        if existing:
            c.execute('''UPDATE records SET
                email=?, patient_id=?, full_name=?, dob=?, gender=?, contact=?,
                address=?, emergency_contact=?, insurance=?, smoking_history=?,
                occupational_exposure=?, family_history=?, previous_conditions=?,
                medications=?, allergies=?, past_surgeries=?, scan_type=?,
                scan_date=?, radiologist_notes=?, ai_output=?, biopsy_performed=?,
                biopsy_result=?, histology=?, genetic_tests=?, cbc=?, cea_level=?,
                other_markers=?, diagnosis_date=?, cancer_stage=?, treatment_type=?,
                treatment_start=?, treatment_response=?, followup_scans=?,
                next_appointment=?, survivorship_notes=?, palliative_notes=?,
                consent_forms=?, doctors=?, referrals=?, medical_notes=?, created_at=?,
                passport=?
                WHERE id=?''', fields + (existing[0],))
        else:
            c.execute('''INSERT INTO records (
                email, patient_id, full_name, dob, gender, contact, address,
                emergency_contact, insurance, smoking_history, occupational_exposure,
                family_history, previous_conditions, medications, allergies, past_surgeries,
                scan_type, scan_date, radiologist_notes, ai_output, biopsy_performed,
                biopsy_result, histology, genetic_tests, cbc, cea_level, other_markers,
                diagnosis_date, cancer_stage, treatment_type, treatment_start,
                treatment_response, followup_scans, next_appointment, survivorship_notes,
                palliative_notes, consent_forms, doctors, referrals, medical_notes, created_at,
                passport
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', fields)

        conn.commit()
        conn.close()
        return jsonify({'message': 'Record saved successfully'}), 200
    except Exception as e:
        logger.error(f"Error saving record: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/records', methods=['GET'])
def get_records():
    try:
        email = request.args.get('email')
        search = request.args.get('search', '')
        patient_id = request.args.get('patientId', '')

        if not email:
            return jsonify({'error': 'Email is required'}), 400

        conn = sqlite3.connect('lunny.db')
        c = conn.cursor()

        if patient_id:
            c.execute('SELECT * FROM records WHERE email = ? AND patient_id = ?', (email, patient_id))
        else:
            query = 'SELECT * FROM records WHERE email = ?'
            params = [email]
            if search:
                query += ' AND (full_name LIKE ? OR patient_id LIKE ?)'
                params.extend([f'%{search}%', f'%{search}%'])
            c.execute(query, params)

        rows = c.fetchall()
        conn.close()

        records = []
        for row in rows:
            record = {
                'patientId': row[2],
                'fullName': row[3],
                'dob': row[4],
                'gender': row[5],
                'contactInfo': row[6],
                'address': row[7],
                'emergencyContact': row[8],
                'insuranceInfo': row[9],
                'smokingHistory': row[10],
                'occupationalExposure': row[11],
                'familyHistory': row[12],
                'previousConditions': row[13],
                'currentMedications': row[14],
                'allergies': row[15],
                'pastSurguries': row[16],
                'scanType': row[17],
                'scanDate': row[18],
                'radiologistNotes': row[19],
                'aiOutput': row[20],
                'biopsyPerformed': row[21],
                'biopsyResult': row[22],
                'histology': row[23],
                'geneticTests': row[24],
                'cbc': row[25],
                'ceaLevel': row[26],
                'otherMarkers': row[27],
                'diagnosisDate': row[28],
                'cancerStage': row[29],
                'treatmentType': row[30],
                'treatmentStartDate': row[31],
                'responseToTreatment': row[32],
                'followUpScans': row[33],
                'nextAppointment': row[34],
                'survivorshipCare': row[35],
                'palliativeNotes': row[36],
                'consentForms': row[37],
                'doctorInCharge': row[38],
                'referrals': row[39],
                'medicalNotes': row[40],
                'passport': row[42]
            }
            records.append(record)

        return jsonify({'records': records}), 200
    except Exception as e:
        logger.error(f"Error fetching records: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect():
    try:
        logger.info("Received /detect request")
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400

        logger.info(f"File received: {file.filename}")

        model_path = r"C:\Users\egwao\OneDrive\Desktop\lung_cancer_detection\resnet50_lung_cancer_trained.pth"

        if not os.path.exists(model_path):
            logger.warning("Model file not found, using random result")
            probability = np.random.random()
            result = 'Positive' if probability > 0.5 else 'Negative'
            confidence = f"{probability * 100:.2f}%"
            return jsonify({
                'result': result,
                'confidence': confidence,
                'details': {
                    'probability': probability,
                    'lung_rads': '4B' if result == 'Positive' else '2',
                    'nodule_type': 'Solid'
                },
                'warning': 'Simulated result - model file missing'
            }), 200

        logger.info(f"Loading model: {model_path}")
        session = ort.InferenceSession(model_path)

        logger.info("Preprocessing image")
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        logger.info("Running inference")
        inputs = {session.get_inputs()[0].name: img_array}
        outputs = session.run(None, inputs)
        probability = float(outputs[0][0][1])
        result = 'Positive' if probability > 0.5 else 'Negative'
        confidence = f"{probability * 100:.2f}%"

        logger.info(f"Result: {result} ({confidence})")

        return jsonify({
            'result': result,
            'confidence': confidence,
            'details': {
                'probability': probability,
                'lung_rads': '4B' if result == 'Positive' else '2',
                'nodule_type': 'Solid'
            }
        }), 200

    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)