from flask import Flask, request, jsonify, render_template, redirect, session, url_for, make_response
import requests
import numpy as np
import librosa
import tensorflow as tf
import pickle
from werkzeug.utils import secure_filename
import os
from pydub import AudioSegment
from datetime import timedelta, datetime
import random
import string
import time

app = Flask(__name__)
app.secret_key = '624512'  # Change this to a random secret key

# Load the trained model and label encoder
model = tf.keras.models.load_model('res_model.h5')
with open('labels.pkl', 'rb') as f:
    lb = pickle.load(f)

# Define the folder to save uploaded audio files
UPLOAD_FOLDER = r'E://data//tested audios'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Convert audio to .wav format
def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_file_path = file_path.rsplit('.', 1)[0] + '.wav'
    audio.export(wav_file_path, format='wav')
    return wav_file_path

# Preprocess and extract MFCC features for Conv2D model
def extract_features_conv2d(audio_file):
    y, sr = librosa.load(audio_file, sr=44100, duration=2.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    if mfccs.shape[1] > 216:
        mfccs = mfccs[:, :216]
    elif mfccs.shape[1] < 216:
        mfccs = np.pad(mfccs, ((0, 0), (0, 216 - mfccs.shape[1])), mode='constant')
    mfccs = mfccs.reshape(mfccs.shape[0], mfccs.shape[1], 1)
    return mfccs[np.newaxis, ...]

@app.route('/')
def index():
    return render_template('new.html')

# Spotify API Credentials
CLIENT_ID = '428bbcecfbdd4e22a0e1eec5adeed462'
CLIENT_SECRET = 'afd41f0466b74180a73cdc62d6313d24'
REDIRECT_URI = 'http://127.0.0.1:5000/callback'
# Authorize the user, handle token expiration and refresh
def refresh_token(refresh_token):
    token_url = 'https://accounts.spotify.com/api/token'
    payload = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }
    response = requests.post(token_url, data=payload)
    token_info = response.json()
    if response.status_code == 200:
        session['access_token'] = token_info.get('access_token')
        if 'refresh_token' in token_info:
            session['refresh_token'] = token_info['refresh_token']
        #session['expires_at'] = (datetime.utcnow() + timedelta(seconds=token_info.get('expires_in'))).timestamp()
        session['expires_at'] = (datetime.now(datetime.timezone.utc) + timedelta(seconds=1)).timestamp()
        response = make_response(redirect(url_for('index')))

        return True
    else:
        return False
@app.route('/authorize')
def authorize():
    if 'access_token' in session and session.get('expires_at', 0) > time.time():
        return redirect(url_for('index'))
    if 'refresh_token' in session:
        if refresh_token(session['refresh_token']):
            return redirect(url_for('index'))
    state = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    session['oauth_state'] = state
    scopes = 'user-read-private user-read-email streaming user-read-playback-state user-modify-playback-state'
    auth_url = f'https://accounts.spotify.com/authorize?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope={scopes}&state={state}'
    return redirect(auth_url)

@app.route('/callback')
def callback():
    state = request.args.get('state')
    if state != session.get('oauth_state'):
        return "Invalid state parameter", 400
    code = request.args.get('code')
    token_url = 'https://accounts.spotify.com/api/token'
    payload = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }
    response = requests.post(token_url, data=payload)
    token_info = response.json()
    if response.status_code != 200:
        return f"Error fetching access token: {response.json()}", 500
    session['access_token'] = token_info.get('access_token')
    session['refresh_token'] = token_info.get('refresh_token')
    expires_in = token_info.get('expires_in')
    session['expires_at'] = (datetime.utcnow() + timedelta(seconds=expires_in)).timestamp()
    response = make_response(redirect(url_for('index')))
    response.set_cookie('login', 'true', max_age=3600)
    return response

# Refresh access token using refresh token



# Callback to handle authorization and token storage


# Predict and redirect based on Spotify premium status
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    if not file_path.endswith('.wav'):
        file_path = convert_to_wav(file_path)
    features = extract_features_conv2d(file_path)
    prediction = model.predict(features)
    emotion = lb.inverse_transform([np.argmax(prediction)])
    access_token = session.get('access_token')
    return redirect(url_for('check_premium', emotion=emotion[0], access_token=access_token))

# Check Spotify premium status and redirect accordingly
@app.route('/check-premium', methods=['GET'])
def check_premium():
    access_token = session.get('access_token')
    detected_emotion = request.args.get('emotion')
    if not access_token:
        return "Error: Access token is missing", 400
    profile = get_spotify_profile(access_token)
    if profile is None:
        return "Error: Unable to fetch user profile", 500
    if profile.get('product') == 'premium':
        return redirect(url_for('result', access_token=access_token, emotion=detected_emotion))
    else:
        return redirect(url_for('normal', access_token=access_token, emotion=detected_emotion))

# Get Spotify user profile
def get_spotify_profile(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get("https://api.spotify.com/v1/me", headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

# Normal user page
@app.route('/normal', methods=['GET'])
def normal():
    emotion = request.args.get('emotion')
    access_token = request.args.get('access_token')
    return render_template('normal.html', emotion=emotion, access_token=access_token)

# Premium user page
@app.route('/result')
def result():
    emotion = request.args.get('emotion')
    access_token = session.get('access_token')
    return render_template('result.html', emotion=emotion, access_token=access_token)

# Handle user logout and optionally revoke token
@app.route('/logout')
def logout():
    session.clear()
    response = make_response(redirect(url_for('index')))
    response.set_cookie('login', '', expires=0)
    return response

if __name__ == '__main__':
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=8)
    app.run(debug=True, port=5000)
