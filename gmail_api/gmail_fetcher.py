import streamlit as st
import os
import pickle
import base64
import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import google.auth.exceptions
import re

# ----------------------------
# Constants and Setup
# ----------------------------
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

# Load the trained model and vectorizer
with open('ml/phishing_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('ml/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# ----------------------------
# Authenticate and Connect to Gmail
# ----------------------------
def authenticate_gmail():
    creds = None
    
    # Load token if it exists
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    # If no token or expired, log in again
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=7777)
            except google.auth.exceptions.DefaultCredentialsError:
                st.error("Missing or invalid credentials.json")
                return None

        # Save the new token
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)
    return service


# ----------------------------
# Fetch Emails from Gmail
# ----------------------------
def fetch_emails(service, max_results=10):
    messages = service.users().messages().list(userId='me', maxResults=max_results).execute().get('messages', [])
    email_texts = []

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        payload = msg_data.get('payload', {})
        headers = payload.get('headers', [])
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '(No Subject)')

        # Get the body
        parts = payload.get('parts', [])
        if parts:
            for part in parts:
                if part.get('mimeType') == 'text/plain':
                    data = part['body'].get('data', '')
                    decoded = base64.urlsafe_b64decode(data.encode('ASCII')).decode('utf-8', errors='ignore')
                    email_texts.append((subject, decoded))
                    break
        else:
            body = payload.get('body', {}).get('data', '')
            if body:
                decoded = base64.urlsafe_b64decode(body.encode('ASCII')).decode('utf-8', errors='ignore')
                email_texts.append((subject, decoded))

    return email_texts


# ----------------------------
# Streamlit UI
# ----------------------------

def classify_emails(emails):
    df = pd.DataFrame(emails, columns=['Subject', 'Body'])
    X_transformed = vectorizer.transform(df['Body'])
    
    # Get probabilities and apply threshold manually
    y_probs = model.predict_proba(X_transformed)[:, 1]
    threshold = 0.6
    df['Prediction'] = (y_probs >= threshold).astype(int)
    df['Prediction'] = df['Prediction'].map({0: 'Phishing Email', 1: 'Safe Email'})
    
    return df


st.title("Gmail Phishing Detector")

if st.button("Authenticate and Fetch Emails"):
    service = authenticate_gmail()
    if service:
        st.success("Authenticated successfully!")
        emails = fetch_emails(service, max_results=10)
        if emails:
            result_df = classify_emails(emails)
            st.write("### Classification Results")
            st.dataframe(result_df)
        else:
            st.warning("No emails found.")
