import cv2 as cv
import numpy as np
import mediapipe as mp

# Инициализация MediaPipe для отслеживания позы
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

progress_squat = 0  # Прогресс приседаний
progress_curl_right = 0  # Прогресс сгибаний для правой руки
progress_curl_left = 0  # Прогресс сгибаний для левой руки
progress_raise_right = 0  # Прогресс махов для правой руки
progress_raise_left = 0  # Прогресс махов для левой руки


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def draw_vertical_progress_bar(frame, progress, x, y, width=20, height=200):
    """Рисует вертикальный прогресс-бар."""
    cv.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)  # Граница
    fill_height = int(height * (progress / 100))
    cv.rectangle(frame, (x, y + height - fill_height), (x + width, y + height), (0, 255, 0), -1)  # Заполнение


def track_squats(frame, landmarks):
    """Отслеживание приседаний с прогресс-баром."""
    global progress_squat

    if landmarks[mp_pose.PoseLandmark.RIGHT_HIP] and landmarks[mp_pose.PoseLandmark.RIGHT_KNEE] and landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]:
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
        angle = calculate_angle(hip, knee, ankle)

        if angle <= 80:
            progress_squat = 100
        elif angle >= 170:
            progress_squat = 0
        else:
            progress_squat = int(100 - ((angle - 80) * (100 / (170 - 80))))  # Линейная интерполяция

    draw_vertical_progress_bar(frame, progress_squat, 50, 50)

def track_bicep_curls(frame, landmarks):
    """Отслеживание бицепсовых сгибаний для обеих рук."""
    global progress_curl_right, progress_curl_left

    # Правая рука
    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER] and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW] and landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]:
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
        angle = calculate_angle(shoulder, elbow, wrist)

        progress_curl_right = max(0, min(100, int(100 - (angle - 90) * (100 / 90)))) if 90 <= angle < 180 else (
            100 if angle <= 90 else 0)

    # Левая рука
    if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER] and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW] and landmarks[mp_pose.PoseLandmark.LEFT_WRIST]:
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        angle = calculate_angle(shoulder, elbow, wrist)

        progress_curl_left = max(0, min(100, int(100 - (angle - 90) * (100 / 90)))) if 90 <= angle < 180 else (
            100 if angle <= 90 else 0)

    draw_vertical_progress_bar(frame, progress_curl_right, 100, 50)
    draw_vertical_progress_bar(frame, progress_curl_left, 130, 50)


def track_lateral_raises(frame, landmarks):
    """Отслеживание махов гантелями для обеих рук."""
    global progress_raise_right, progress_raise_left

    # Правая рука
    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER] and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]:
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        angle = calculate_angle(hip, shoulder, elbow)

        progress_raise_right = int((angle / 90) * 100) if 0 <= angle <= 90 else 0

    # Левая рука
    if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER] and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]:
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        angle = calculate_angle(hip, shoulder, elbow)

        progress_raise_left = int((angle / 90) * 100) if 0 <= angle <= 90 else 0

    draw_vertical_progress_bar(frame, progress_raise_right, 180, 50)
    draw_vertical_progress_bar(frame, progress_raise_left, 210, 50)
