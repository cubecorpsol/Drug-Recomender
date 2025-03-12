from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
app = Flask(__name__)
app.secret_key = os.urandom(24)
load_dotenv()
# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
db = SQLAlchemy(app)


# Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True  # Fixed typo
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
mail = Mail(app)

# File Paths
csv_path = './data/DrugData_final.csv'
model_path = './models/gbm_drug_recommendation_model.pkl'
encoder_path = './models/label_encoder(2).pkl'

# Load Model and Data
try:
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    drug_data = pd.read_csv(csv_path, encoding='latin-1')
except Exception as e:
    print("Error loading model or data:", e)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)  # Increased size for hashed password
    verified = db.Column(db.Boolean, default=False)

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if not user:
            flash("No account found with this email.", "danger")
            return redirect(url_for('login'))

        if not check_password_hash(user.password, password):
            flash("Incorrect password. Try again.", "danger")
            return redirect(url_for('login'))

        if not user.verified:
            flash("Verify your email before logging in.", "warning")
            return redirect(url_for('login'))

        session['user_id'] = user.id
        flash("Login successful!", "success")
        return redirect(url_for('dashboard'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        phone = request.form.get('phone')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(email=email).first():
            flash("Email is already registered.", "danger")
            return redirect(url_for('register'))

        if User.query.filter_by(phone=phone).first():
            flash("Phone number is already registered.", "danger")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')  # Fixed hashing
        new_user = User(name=name, age=age, gender=gender, phone=phone, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        send_verification_email(new_user)
        flash("Registration successful! Check your email to verify your account.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

def send_verification_email(user):
    """ Sends an email verification link to the user. """
    token_link = url_for('verify_email', user_id=user.id, _external=True)
    msg = Message("Verify Your Email", sender=app.config['MAIL_USERNAME'], recipients=[user.email])
    msg.body = f"Click the link to verify your account: {token_link}"
    mail.send(msg)

@app.route('/verify/<int:user_id>')
def verify_email(user_id):
    user = User.query.get(user_id)
    if user:
        user.verified = True
        db.session.commit()
        flash("Your email has been verified.", "success")
    else:
        flash("Invalid verification link.", "danger")

    return redirect(url_for('login'))

def load_symptoms_and_dosage():
    try:
        df = pd.read_csv(csv_path, encoding='latin-1')
        if 'symptoms' in df.columns and 'dosageform' in df.columns:
            symptoms_list = df[['symptoms', 'dosageform']].dropna().drop_duplicates()
            symptoms_dict = dict(zip(symptoms_list['symptoms'], symptoms_list['dosageform']))
            return sorted(symptoms_list['symptoms'].tolist()), symptoms_dict
    except Exception as e:
        print("Error loading symptoms and dosage:", str(e))
    return [], {}

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    symptoms, dosage_mapping = load_symptoms_and_dosage()
    return render_template('dashboard.html', symptoms=symptoms, dosage_mapping=dosage_mapping)

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    symptoms = request.form.get('symptoms')
    dosage_form = request.form.get('dosage_form')
    age_group = request.form.get('age_group')

    if not symptoms or not dosage_form or not age_group:
        flash("Please select all fields before submitting.", "warning")
        return redirect(url_for('dashboard'))

    try:
        input_data = pd.DataFrame({'symptoms': [symptoms], 'dosageform': [dosage_form]})
        predicted_label = model.predict(input_data)
        recommended_drug = label_encoder.inverse_transform(predicted_label)[0]

        if recommended_drug not in drug_data['Drug Name'].values:
            flash("No suitable drug found.", "warning")
            return redirect(url_for('dashboard'))

        drug_info = drug_data[drug_data['Drug Name'] == recommended_drug].iloc[0]

        if age_group not in drug_info:
            flash("Age group not found in dataset.", "danger")
            return redirect(url_for('dashboard'))

        dosage = drug_info[age_group]
        alternatives = [drug_info.get('Alternative 1', 'None'), drug_info.get('Alternative 2', 'None')]

        return render_template('recommendation.html', 
                               drug=recommended_drug,
                               dosage=dosage,
                               alternatives=alternatives)
    except Exception as e:
        flash(f"Error in recommendation: {str(e)}", "danger")
        return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# Run the Application
if __name__ == '__main__':
    with app.app_context():
        db.create_all() 
    app.run(debug=True) 
