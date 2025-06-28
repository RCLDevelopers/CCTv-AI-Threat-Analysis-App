import cv2
import numpy as np
import time
import os
from datetime import datetime
from decouple import config
from twilio.rest import Client
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from imutils.video import VideoStream

# AI Model Imports
from ultralytics import YOLO
import mediapipe as mp
from transformers import CLIPProcessor, CLIPModel
import torch

class AISecurityCamera:
    def __init__(self):
        # Initialize camera
        self.vs = VideoStream(src=0).start()
        time.sleep(2.0)
        
        # Initialize AI models
        self.yolo_model = YOLO('yolov8n.pt')  # Weapon detection
        self.pose_model = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load configuration
        self.email_config = {
            'host': config('EMAIL_HOST'),
            'port': config('EMAIL_PORT', cast=int),
            'user': config('EMAIL_USER'),
            'password': config('EMAIL_PASSWORD'),
            'recipient': config('EMAIL_RECIPIENT')
        }
        
        # Twilio setup
        self.twilio_client = Client(config('TWILIO_ACCOUNT_SID'), config('TWILIO_AUTH_TOKEN'))
        self.twilio_number = config('TWILIO_PHONE_NUMBER')
        self.my_number = config('YOUR_PHONE_NUMBER')
        
        # Security parameters
        self.motion_threshold = config('MOTION_THRESHOLD', cast=int)
        self.min_area = config('MIN_AREA', cast=int)
        self.alert_cooldown = 300  # 5 minutes between alerts
        
        # Threat categories
        self.threat_labels = [
            "weapon", "gun", "knife", "mask", "breaking in", 
            "burglary", "vandalism", "suspicious activity"
        ]
        
        # Pose analysis parameters
        self.suspicious_poses = {
            'crouching': (0.7, 1.5),  # Knee-to-hip ratio range
            'reaching': (1.5, 3.0),   # Arm extension ratio
            'loitering': (15, 30)      # Time threshold in seconds
        }

    def detect_objects(self, frame):
        """Detect weapons and suspicious objects using YOLOv8"""
        results = self.yolo_model(frame)
        detected_objects = []
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                confidence = float(box.conf[0])
                
                if confidence > 0.5:  # Only consider confident detections
                    detected_objects.append({
                        'label': label,
                        'confidence': confidence,
                        'box': box.xyxy[0].tolist()
                    })
        
        return detected_objects

    def analyze_pose(self, frame):
        """Analyze human pose for suspicious behavior"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_model.process(rgb_frame)
        pose_analysis = {'suspicious': False, 'actions': []}
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Check for crouching
            left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            knee_hip_ratio = abs(left_knee.y - left_hip.y)
            
            if self.suspicious_poses['crouching'][0] < knee_hip_ratio < self.suspicious_poses['crouching'][1]:
                pose_analysis['actions'].append('crouching')
                pose_analysis['suspicious'] = True
            
            # Check for reaching
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            reach_ratio = abs(left_wrist.x - left_shoulder.x)
            
            if reach_ratio > self.suspicious_poses['reaching'][0]:
                pose_analysis['actions'].append('reaching')
                pose_analysis['suspicious'] = True
        
        return pose_analysis

    def analyze_scene(self, frame):
        """Use CLIP to understand the scene context"""
        inputs = self.clip_processor(
            text=self.threat_labels,
            images=frame, 
            return_tensors="pt", 
            padding=True
        )
        
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).detach().numpy()
        
        threat_results = []
        for i, label in enumerate(self.threat_labels):
            if probs[0][i] > 0.3:  # Probability threshold
                threat_results.append({
                    'label': label,
                    'confidence': float(probs[0][i])
                })
        
        return threat_results

    def is_threat(self, frame):
        """Combine all AI analyses to determine if there's a threat"""
        threat_detected = False
        threat_description = []
        
        # Object detection
        objects = self.detect_objects(frame)
        weapon_objects = [obj for obj in objects if obj['label'] in ['knife', 'gun']]
        
        if weapon_objects:
            threat_detected = True
            for obj in weapon_objects:
                threat_description.append(
                    f"{obj['label']} detected (confidence: {obj['confidence']:.2f})"
                )
        
        # Pose analysis
        pose = self.analyze_pose(frame)
        if pose['suspicious']:
            threat_detected = True
            threat_description.append(
                f"Suspicious poses detected: {', '.join(pose['actions'])}"
            )
        
        # Scene understanding
        scene = self.analyze_scene(frame)
        if scene:
            threat_detected = True
            for item in scene:
                threat_description.append(
                    f"Scene indicates {item['label']} (confidence: {item['confidence']:.2f})"
                )
        
        return threat_detected, threat_description

    def send_alerts(self, frame, threat_description):
        """Send alerts with threat analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"threat_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        
        # Prepare threat description text
        description_text = "SECURITY THREAT DETECTED!\n\n"
        description_text += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        description_text += "Threat Analysis:\n- " + "\n- ".join(threat_description)
        
        # Send email
        self.send_email_alert(image_path, description_text)
        
        # Send WhatsApp
        self.send_whatsapp_alert(image_path, description_text)
        
        # Clean up image file
        os.remove(image_path)

    def send_email_alert(self, image_path, description):
        msg = MIMEMultipart()
        msg['From'] = self.email_config['user']
        msg['To'] = self.email_config['recipient']
        msg['Subject'] = "SECURITY ALERT: Threat Detected!"
        
        msg.attach(MIMEText(description, 'plain'))
        
        with open(image_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
            msg.attach(img)
        
        try:
            server = smtplib.SMTP(self.email_config['host'], self.email_config['port'])
            server.starttls()
            server.login(self.email_config['user'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            print("Email alert sent successfully")
        except Exception as e:
            print(f"Failed to send email: {e}")

    def send_whatsapp_alert(self, image_path, description):
        try:
            # First send text
            self.twilio_client.messages.create(
                body=description,
                from_=self.twilio_number,
                to=self.my_number
            )
            
            # Then send image (WhatsApp allows separate messages)
            with open(image_path, 'rb') as f:
                self.twilio_client.messages.create(
                    media_url=[image_path],
                    from_=self.twilio_number,
                    to=self.my_number
                )
            
            print("WhatsApp alert sent successfully")
        except Exception as e:
            print(f"Failed to send WhatsApp message: {e}")

    def run(self):
        last_alert_time = 0
        loitering_start_time = None
        
        while True:
            frame = self.vs.read()
            if frame is None:
                break
                
            frame = cv2.resize(frame, (640, 480))
            
            # Check for threats
            threat_detected, threat_description = self.is_threat(frame)
            
            current_time = time.time()
            if threat_detected:
                if current_time - last_alert_time > self.alert_cooldown:
                    self.send_alerts(frame, threat_description)
                    last_alert_time = current_time
            
            # Check for loitering (prolonged presence without specific threats)
            if len(self.detect_objects(frame)) > 0:  # If people detected
                if loitering_start_time is None:
                    loitering_start_time = current_time
                elif current_time - loitering_start_time > self.suspicious_poses['loitering'][1]:
                    threat_detected = True
                    threat_description = ["Person loitering for extended period"]
                    if current_time - last_alert_time > self.alert_cooldown:
                        self.send_alerts(frame, threat_description)
                        last_alert_time = current_time
                        loitering_start_time = None
            else:
                loitering_start_time = None
            
            # Display (optional)
            cv2.imshow("AI Security Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Loading AI security system...")
    camera = AISecurityCamera()
    camera.run()