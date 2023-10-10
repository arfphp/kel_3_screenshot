import tkinter as tk
from tkinter import Button
import cv2
import numpy as np
import mediapipe as mp
import os
import threading
import time

# Inisialisasi variabel global
screenshot_delay = 5  # Cooldown 5 detik
peace_threshold = 100  # Threshold untuk mengambil gambar saat mendeteksi "peace"

# Inisialisasi variabel cooldown_count
cooldown_count = screenshot_delay

# Fungsi untuk memuat model deteksi objek
def load_object_detection_model():
    net = cv2.dnn.readNetFromCaffe('ssd_files/deploy.prototxt', 'ssd_files/res10_300x300_ssd_iter_140000.caffemodel')
    return net

# Fungsi untuk melakukan deteksi objek pada suatu frame
def perform_object_detection(frame, net):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    detected_objects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            object_class = int(detections[0, 0, i, 1])
            detected_objects.append((object_class, confidence, detections[0, 0, i, 3:7]))

    return detected_objects

# Fungsi untuk deteksi gestur tangan "peace"
def is_peace_gesture(landmarks):
    # Mengukur jarak antara dua jari tengah
    index_finger = landmarks[8]
    middle_finger = landmarks[12]
    distance = np.linalg.norm(np.array([index_finger.x, index_finger.y]) - np.array([middle_finger.x, middle_finger.y]))

    # Jika jarak lebih pendek dari threshold, deteksi "peace"
    return distance < peace_threshold

# Fungsi untuk mengambil gambar
def take_screenshot(frame, screenshot_count):
    # Simpan gambar saat ini sebagai file JPEG
    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    image_filename = os.path.join(screenshot_dir, f"screenshot_{screenshot_count}.jpg")
    cv2.imwrite(image_filename, frame)
    print(f"Screenshot disimpan sebagai {image_filename}")

# Fungsi untuk menjalankan deteksi objek dan tangan secara berkelanjutan
def detect_objects_and_hand():
    global cooldown_count  # Menggunakan variabel global cooldown_count
    # Untuk mengakses kamera pada komputer menggunakan OpenCV
    cap = cv2.VideoCapture(0)
    # Turunkan resolusi kamera
    cap.set(3, 640)
    cap.set(4, 480)
    net = load_object_detection_model()

    # Membuat objek mediapipe untuk deteksi tangan
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Daftar kelas objek yang mungkin terdeteksi
    object_classes = ["Unknown", "Person"]
    # Warna acak untuk setiap kelas objek
    COLORS = np.random.uniform(0, 255, size=(len(object_classes), 3))

    selfie_counter = 0
    screenshot_count = 1

    def take_screenshot_thread():
        nonlocal selfie_counter, screenshot_count
        while True:
            if selfie_counter >= screenshot_delay:
                ret, frame = cap.read()
                if not ret:
                    continue
                take_screenshot(frame, screenshot_count)
                screenshot_count += 1
                selfie_counter = 0

    # Buat dan jalankan thread untuk pengambilan screenshot
    screenshot_thread = threading.Thread(target=take_screenshot_thread)
    screenshot_thread.daemon = True
    screenshot_thread.start()

    while True:
        ret, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                for landmark in landmarks.landmark:
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                if is_peace_gesture(landmarks.landmark):
                    countdown = cooldown_count
                    if countdown > 0:
                        countdown_text = f"Countdown: {countdown}"
                        cv2.putText(frame, countdown_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        take_screenshot(frame, screenshot_count)
                        screenshot_count += 1
                        cooldown_count = screenshot_delay

        detected_objects = perform_object_detection(frame, net)

        for obj_class, confidence, bbox in detected_objects:
            if obj_class < len(object_classes):
                label = object_classes[obj_class]
                color = COLORS[obj_class]

                if label != "Person":
                    label = "Unknown"
                h, w = frame.shape[:2]
                startX, startY, endX, endY = bbox * np.array([w, h, w, h])
                box_width = int(endX - startX)
                box_height = int(endY - startY)
                cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), color, 2)
                y = int(startY) - 15 if int(startY) - 15 > 15 else int(startY) + 15
                confidence_percentage = f"{confidence * 100:.2f}%"
                cv2.putText(frame, f"{label}: {confidence_percentage}", (int(startX), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Menampilkan frame dengan objek yang terdeteksi
        cv2.imshow('Object Detection', frame)

        selfie_counter += 1

        if cv2.waitKey(1) == ord('c'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Mulai aplikasi
app = tk.Tk()
app.title("Object Detection")

# tombol untuk memulai deteksi objek dan tangan
start_button = tk.Button(app, text="Start Detection", command=detect_objects_and_hand)
start_button.pack(pady=10)

# loop utama antarmuka grafis
app.mainloop()
