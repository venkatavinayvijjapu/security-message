#  This code send the text message and the detected faces also.

import torch
import numpy as np
import cv2
from time import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
import os

import supervision as sv
from ultralytics import YOLO
from email_data import password, from_email, to_email

server = smtplib.SMTP('smtp.gmail.com: 587')
server.starttls()
server.login(from_email, password)

class ObjectDetection:
    def __init__(self, capture_index) -> None:
        self.capture_index = capture_index
        self.email_sent = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device:", self.device)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)

    def load_model(self):
        model = YOLO('yolov8n.pt')
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            if class_id == 0:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
        detections = sv.Detections.from_ultralytics(results[0])
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        return frame, class_ids

    def send_email_with_image(self, to_email, from_email, image_path, people_detected=1):
        message = MIMEMultipart()
        message['From'] = from_email
        message['To'] = to_email
        message['Subject'] = "Security Alert"

        message.attach(MIMEText(f'Check Check - {people_detected} has come into the camera', 'plain'))

        with open(image_path, 'rb') as img_file:
            img = MIMEImage(img_file.read())
            img.add_header('Content-Disposition', 'attachment', filename='detected_person.jpg')
            message.attach(img)

        server.sendmail(from_email, to_email, message.as_string())

    def capture_image(self, frame):
        temp_image_path = 'temp_detected_person.jpg'
        cv2.imwrite(temp_image_path, frame)
        return temp_image_path

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
      
        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            frame, class_ids = self.plot_bboxes(results, frame)
            
            if len(class_ids) > 0:
                if not self.email_sent:
                    temp_image_path = self.capture_image(frame)
                    self.send_email_with_image(to_email, from_email, temp_image_path, len(class_ids))
                    self.email_sent = True
                    os.remove(temp_image_path)
            else:
                self.email_sent = False

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('YOLOv8 Detection', frame)

            frame_count += 1

            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        server.quit()

detector = ObjectDetection(capture_index=0)
detector()
