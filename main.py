import sys
import time

from PyQt5.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout,)
from PyQt5.QtGui import (QPixmap, QImage, QColor, QPalette,)
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
import pyttsx3
from postprocessing import *

class GUIYoloV8(QWidget):
    CONFIDENCE_THRESHOLD = 80

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GUIYoloV8")
        self.setFixedSize(1400, 900)  # Set a fixed size for the window
        self.layout = QVBoxLayout()
        self.default_image_path = "interface.jpeg"

        # Initialize the text-to-speech engine
        self.tts_engine = pyttsx3.init()

        self.last_spoken_sign = None
        self.last_spoken_time = time.time()

        # Title Label
        title_label = QLabel("Traffic Sign Detection with YOLOv8")
        title_label.setAlignment(Qt.AlignHCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-top: 10px; margin-bottom: 10px;")
        self.layout.addWidget(title_label)

        # Result Display
        self.result_label = QLabel()
        self.result_label.setStyleSheet("border: 1px solid rgb(127, 129, 130); margin: 10px; padding: 10px; border-radius: 10px;")
        self.result_label.setFixedHeight(620)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

        # Set the default image in the result label
        default_image = QPixmap(self.default_image_path)
        self.result_label.setPixmap(default_image.scaled(self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))

        # Detected Class Label
        self.class_label = QLabel()
        self.class_label.setAlignment(Qt.AlignCenter)
        self.class_label.setStyleSheet("border: 1px solid rgb(127, 129, 130); font-size: 16px; font-weight: bold; margin: 10px; padding: 10px; border-radius: 10px; background-color: rgb(25, 25, 25);")
        # self.class_label.setFixedHeight(80)
        self.layout.addWidget(self.class_label)

        # Create a QHBoxLayout for button layout
        selection_button_layout = QHBoxLayout()

        # Select Image Button
        image_button = QPushButton("Select Image")
        image_button.setStyleSheet("font-size: 18px; font-weight: bold; color:rgb(247, 247, 247) ;padding: 5px; margin: 5px; border-radius: 10px; background-color: rgb(127, 129, 130);")
        image_button.clicked.connect(self.select_image)
        selection_button_layout.addWidget(image_button)
        # self.layout.addWidget(image_button)

        # Select Video Button
        video_button = QPushButton("Select Video")
        video_button.setStyleSheet("font-size: 18px; font-weight: bold; color:rgb(247, 247, 247) ; padding: 5px; margin: 5px; border-radius: 10px; background-color: rgb(127, 129, 130);")
        video_button.clicked.connect(self.select_video)
        selection_button_layout.addWidget(video_button)
        # self.layout.addWidget(video_button)

        # Start Webcam Button
        self.start_webcam_button = QPushButton("Start Webcam")
        self.start_webcam_button.setStyleSheet("font-size: 18px; font-weight: bold; color:rgb(247, 247, 247) ; padding: 5px; margin: 5px; border-radius: 10px; background-color: rgb(127, 129, 130);")
        self.start_webcam_button.clicked.connect(self.start_webcam)
        selection_button_layout.addWidget(self.start_webcam_button)

        # Add button layout to the main layout
        self.layout.addLayout(selection_button_layout)

        end_button_layout = QHBoxLayout() 

        # Stop Button
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("font-size: 18px; font-weight: bold; color:rgb(247, 247, 247) ; padding: 5px; margin: 5px; border-radius: 10px; background-color: rgb(127, 129, 130);")
        self.stop_button.clicked.connect(self.stop_prediction)
        # self.stop_button.setEnabled(False)
        end_button_layout.addWidget(self.stop_button)

        # Close Application Button
        close_button = QPushButton("Close")
        close_button.setStyleSheet("font-size: 18px; font-weight: bold; color:rgb(247, 247, 247) ; padding: 5px; margin: 5px; border-radius: 10px; background-color: rgb(127, 129, 130);")
        close_button.clicked.connect(self.close)
        end_button_layout.addWidget(close_button)

        self.layout.addLayout(end_button_layout)

        self.setLayout(self.layout)

        self.video_file = None
        self.model = None
        self.class_list = None
        self.scale_show = 100
        self.video_capture = None

    def select_image(self):
        image_file, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if image_file:
            self.video_file = None
            self.start_prediction_for_image(image_file)

    def select_video(self):
        video_file, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if video_file:
            self.video_file = video_file
            self.start_prediction_for_video(video_file)

    def start_webcam(self):
        self.video_file = 0
        self.result_label.setPixmap(QPixmap(self.default_image_path).scaled(self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))
        # self.start_webcam_button.setEnabled(False)
        # self.stop_button.setEnabled(True)
        self.start_prediction_for_video("webcam")   

    def stop_prediction(self):
        self.video_file = None
        # self.stop_button.setEnabled(False)
        # self.start_webcam_button.setEnabled(True)
        self.video_capture.release()
        self.result_label.clear()
        self.class_label.clear()
        self.result_label.setPixmap(QPixmap(self.default_image_path).scaled(self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))

    def start_prediction_for_image(self, file):
        self.model = YOLO("trained-models/best.pt")
        self.class_list = { 0: 'DP.135', 1: 'P.102', 2: 'P.103a', 3: 'P.103b', 4: 'P.103c', 5: 'P.104', 6: 'P.106a', 7: 'P.106b', 8: 'P.107a', 9: 'P.112', 10: 'P.115', 11: 'P.117', 12: 'P.123a', 13: 'P.123b', 14: 'P.124a', 15: 'P.124b', 16: 'P.124c', 17: 'P.125', 18: 'P.127', 19: 'P.128', 20: 'P.130', 21: 'P.131a', 22: 'P.137', 23: 'P.245a', 24: 'R.301c', 25: 'R.301d', 26: 'R.301e', 27: 'R.302a', 28: 'R.302b', 29: 'R.303', 30: 'R.407a', 31: 'R.409', 32: 'R.425', 33: 'R.434', 34: 'S.509a', 35: 'W.201a', 36: 'W.201b', 37: 'W.202a', 38: 'W.202b', 39: 'W.203b', 40: 'W.203c', 41: 'W.205a', 42: 'W.205b', 43: 'W.205d', 44: 'W.207a', 45: 'W.207b', 46: 'W.207c', 47: 'W.208', 48: 'W.209', 49: 'W.210', 50: 'W.219', 51: 'W.221b', 52: 'W.224', 53: 'W.225', 54: 'W.227', 55: 'W.233', 56: 'W.235', 57: 'W.245a' }

        self.video_capture = cv2.VideoCapture(file)

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            # Convert RGBA image to RGB (if required)
            if frame.shape[2] == 4:
                # print("inside")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            results = self.model.predict(frame, verbose=False)
            labeled_img = draw_box(frame, results[0], self.class_list, 2, 1)
            display_img = resize_image(labeled_img, self.scale_show)

            # Convert the image to QImage
            rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Display the image in the QLabel
            pixmap = QPixmap(q_image)
            self.result_label.setPixmap(pixmap.scaled(self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))

            # Update the detected class label
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                labels = []
                for i in range(len(results[0].boxes)):
                    class_id = results[0].boxes.cls[i].item()  # Convert tensor to integer
                    if class_id in self.class_list:
                        class_name = self.class_list[class_id]
                        confidence = results[0].boxes.conf[i] * 100
                        # Add this condition to display label only if confidence is over fixed presentage
                        if confidence > GUIYoloV8.CONFIDENCE_THRESHOLD:
                            label = f"{class_name}: {confidence:.2f}%"
                            labels.append(label)
                label_text = "\n".join(labels)
            else:
                label_text = "No detection"
            self.class_label.setText(label_text)

             # Determine whether to say "No detection" or not based on the file type
            image_selection = file.endswith((".png", ".jpg", ".jpeg"))

            # Process GUI events
            QApplication.processEvents()

            # Output detected signs using speech
            self.speak_detected_signs(results[0].boxes, image_selection)

    def start_prediction_for_video(self, file):
        self.model = YOLO("trained-models/best.pt")
        self.class_list = { 0: 'DP.135', 1: 'P.102', 2: 'P.103a', 3: 'P.103b', 4: 'P.103c', 5: 'P.104', 6: 'P.106a', 7: 'P.106b', 8: 'P.107a', 9: 'P.112', 10: 'P.115', 11: 'P.117', 12: 'P.123a', 13: 'P.123b', 14: 'P.124a', 15: 'P.124b', 16: 'P.124c', 17: 'P.125', 18: 'P.127', 19: 'P.128', 20: 'P.130', 21: 'P.131a', 22: 'P.137', 23: 'P.245a', 24: 'R.301c', 25: 'R.301d', 26: 'R.301e', 27: 'R.302a', 28: 'R.302b', 29: 'R.303', 30: 'R.407a', 31: 'R.409', 32: 'R.425', 33: 'R.434', 34: 'S.509a', 35: 'W.201a', 36: 'W.201b', 37: 'W.202a', 38: 'W.202b', 39: 'W.203b', 40: 'W.203c', 41: 'W.205a', 42: 'W.205b', 43: 'W.205d', 44: 'W.207a', 45: 'W.207b', 46: 'W.207c', 47: 'W.208', 48: 'W.209', 49: 'W.210', 50: 'W.219', 51: 'W.221b', 52: 'W.224', 53: 'W.225', 54: 'W.227', 55: 'W.233', 56: 'W.235', 57: 'W.245a' }

        if file == "webcam":
            self.video_capture = cv2.VideoCapture(0)
        else:
            self.video_capture = cv2.VideoCapture(file)

        frames_to_skip = 10  # Process every 10th frame (change this as needed)
        reduced_scale_percent = 100  # Adjust this value for desired video quality
        frame_count = 0

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame_count += 1
            if self.video_capture:
                if frame_count % frames_to_skip != 0:
                    continue

            # Resize frame to reduce the resolution
            reduced_frame = resize_image(frame, reduced_scale_percent)

            # Convert RGBA image to RGB (if required)
            if reduced_frame.shape[2] == 4:
                # print("inside")
                reduced_frame = cv2.cvtColor(reduced_frame, cv2.COLOR_RGBA2RGB)

            results = self.model.predict(reduced_frame, verbose=False)
            labeled_img = draw_box(reduced_frame, results[0], self.class_list, 1, 0.5)
            display_img = resize_image(labeled_img, self.scale_show)

            # Convert the image to QImage
            rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Display the image in the QLabel
            pixmap = QPixmap(q_image)
            self.result_label.setPixmap(pixmap.scaled(self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))

            # Update the detected class label
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                labels = []
                for i in range(len(results[0].boxes)):
                    class_id = results[0].boxes.cls[i].item()  # Convert tensor to integer
                    if class_id in self.class_list:
                        class_name = self.class_list[class_id]
                        confidence = results[0].boxes.conf[i] * 100
                        # Add this condition to display label only if confidence is over fixed presentage
                        if confidence > GUIYoloV8.CONFIDENCE_THRESHOLD:
                            label = f"{class_name}: {confidence:.2f}%"
                            labels.append(label)
                        else:
                            results[0].boxes
                label_text = "\n".join(labels) if labels else "No high-confidence detection"
            else:
                label_text = "No detection"

            self.class_label.setText(label_text)

             # Determine whether to say "No detection" or not based on the file type
            image_selection = file.endswith((".png", ".jpg", ".jpeg"))

            # Process GUI events
            QApplication.processEvents()

            # Output detected signs using speech
            self.speak_detected_signs(results[0].boxes, image_selection)

    def speak_detected_signs(self, boxes, image_selection):
        if boxes is not None and len(boxes) > 0:
            detected_signs = []
            for i in range(len(boxes)):
                class_id = boxes.cls[i].item()  # Convert tensor to integer
                if class_id in self.class_list:
                    class_name = self.class_list[class_id]
                    confidence = boxes.conf[i] * 100
                    if confidence > GUIYoloV8.CONFIDENCE_THRESHOLD:  # Speak only high-confidence detections
                        detected_signs.append(class_name)
                    
            if detected_signs:
                current_time = time.time()
                if detected_signs != self.last_spoken_sign or current_time - self.last_spoken_time > 5:
                    speech_text = " and ".join(detected_signs)  # Combine multiple detections with "and"
                    self.tts_engine.say(speech_text)
                    self.tts_engine.runAndWait()
                    self.last_spoken_sign = detected_signs
                    self.last_spoken_time = current_time
        else:
            if image_selection:
                self.tts_engine.say("No detection")
                self.tts_engine.runAndWait()

    def closeEvent(self, event):
        self.video_capture.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set custom styles
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(0, 0, 0))  # Set background color
    palette.setColor(QPalette.WindowText, QColor(247, 247, 247))  # Set text color
    app.setPalette(palette)

    gui = GUIYoloV8()
    gui.show()

    sys.exit(app.exec_())
