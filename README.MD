# AI Security Camera System

This project is an AI-powered security camera system that uses computer vision and deep learning to detect weapons, suspicious behavior, and potential threats in real-time video feeds. It sends alerts via email and WhatsApp when a threat is detected.

## Features

- **Weapon Detection:** Uses YOLOv8 to detect weapons (guns, knives) and other suspicious objects.
- **Pose Analysis:** Uses MediaPipe to detect suspicious human poses (crouching, reaching, loitering).
- **Scene Understanding:** Uses OpenAI CLIP to analyze the scene for threat context.
- **Alert System:** Sends alerts with images and threat descriptions via email and WhatsApp (Twilio).
- **Configurable:** All credentials and thresholds are managed via a `.env` file.

---

## Setup Instructions

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Create and Activate a Python Virtual Environment

```sh
python -m venv security-env
# On Windows:
security-env\Scripts\activate
# On macOS/Linux:
source security-env/bin/activate
```

### 3. Install Required Dependencies

```sh
pip install opencv-python numpy python-decouple twilio imutils ultralytics mediapipe torch transformers
```

### 4. Download Pre-trained Models

Run the provided script to download the required models:

```sh
python download_models.py
```

### 5. Configure Environment Variables

Create a `.env` file in the project root (if not already present) and fill in your credentials:

```env
# Email Configuration
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_email_app_password
EMAIL_RECIPIENT=recipient_email@gmail.com

# Twilio (WhatsApp) Configuration
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=whatsapp:+your_twilio_whatsapp_number
YOUR_PHONE_NUMBER=whatsapp:+your_whatsapp_number

# Security Thresholds
MOTION_THRESHOLD=500
MIN_AREA=500
```

**Note:**  
- For Gmail, you may need to use an [App Password](https://support.google.com/accounts/answer/185833) if 2FA is enabled.
- For WhatsApp via Twilio, ensure your Twilio account is set up for WhatsApp messaging.

### 6. Run the AI Security Camera

```sh
python ai_security_camera.py
```

---

## How It Works

- The script captures video from your default webcam.
- It analyzes each frame for weapons, suspicious poses, and scene context.
- If a threat is detected, it sends an alert with an image and description via email and WhatsApp.
- The system enforces a cooldown period between alerts to avoid spamming.

---

## Troubleshooting

- **Camera Not Detected:** Ensure your webcam is connected and not used by another application.
- **Email Not Sending:** Double-check your SMTP credentials and app password.
- **WhatsApp Not Sending:** Verify your Twilio credentials and WhatsApp sandbox setup.
- **Model Download Issues:** Ensure you have a stable internet connection when running `download_models.py`.

---

## File Structure

- [`ai_security_camera.py`](ai_security_camera.py): Main script for the AI security system.
- [`download_models.py`](download_models.py): Script to download required AI models.
- `.env`: Configuration file for credentials and thresholds.
- `yolov8n.pt`: YOLOv8 model weights.
- `security-env/`: Python virtual environment (not required to commit to version control).

---

## Requirements

- Python 3.8+
- Webcam

---

## License

This project is for educational and research purposes.

---

## Credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://mediapipe.dev/)
-