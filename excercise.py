import cv2 as cv
import numpy as np
import mediapipe as mp

# Инициализация MediaPipe для отслеживания позы
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

progress = 0  # Прогресс выполнения упражнения


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def draw_progress_bar(frame, progress, x, y, width=300, height=20):
    cv.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)  # Граница
    cv.rectangle(frame, (x, y), (x + int(width * (progress / 100)), y + height), (0, 255, 0), -1)  # Заполнение


def track_squats(frame, landmarks):
    global progress

    if landmarks[mp_pose.PoseLandmark.RIGHT_HIP] and landmarks[mp_pose.PoseLandmark.RIGHT_KNEE] and landmarks[
        mp_pose.PoseLandmark.RIGHT_ANKLE]:
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
        angle = calculate_angle(hip, knee, ankle)

        progress = max(0, min(100, int(100 - (angle - 90) * (100 / 70)))) if 90 <= angle < 160 else (
            100 if angle <= 90 else 0)

    draw_progress_bar(frame, progress, 50, 50)


def track_bicep_curls(frame, landmarks):
    global progress

    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER] and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW] and landmarks[
        mp_pose.PoseLandmark.RIGHT_WRIST]:
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
        angle = calculate_angle(shoulder, elbow, wrist)

        progress = max(0, min(100, int(100 - (angle - 90) * (100 / 90)))) if 90 <= angle < 180 else (
            100 if angle <= 90 else 0)

    draw_progress_bar(frame, progress, 50, 100)


def track_lateral_raises(frame, landmarks):
    global progress

    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER] and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]:
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        angle = calculate_angle(hip, shoulder, elbow)

        progress = int((angle / 90) * 100) if 0 <= angle <= 90 else 0

    draw_progress_bar(frame, progress, 50, 150)
