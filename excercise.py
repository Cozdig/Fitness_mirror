import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque

# Инициализация MediaPipe для отслеживания позы
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Буферы для скользящего среднего
N = 5
progress_squat_buffer = deque(maxlen=N)
progress_curl_right_buffer = deque(maxlen=N)
progress_curl_left_buffer = deque(maxlen=N)
progress_raise_right_buffer = deque(maxlen=N)
progress_raise_left_buffer = deque(maxlen=N)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def smooth_progress(buffer, new_value):
    buffer.append(new_value)
    return int(np.mean(buffer))

def draw_vertical_progress_bar(frame, progress, x, y, width=20, height=200):
    """Рисует вертикальный прогресс-бар."""
    cv.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)  # Граница
    fill_height = int(height * (progress / 100))
    cv.rectangle(frame, (x, y + height - fill_height), (x + width, y + height), (0, 255, 0), -1)  # Заполнение


def track_squats(frame, landmarks):
    """Отслеживание фронтальных приседаний с обновленным расчетом угла."""

    def get_squat_progress(hip, knee, ankle):
        """Вычисляет прогресс приседаний на основе угла бедра относительно вертикали."""
        hip, knee, ankle = np.array(hip), np.array(knee), np.array(ankle)

        # Вычисляем угол между бедром и голенью (Y-координаты)
        angle = calculate_angle(hip, knee, ankle)

        # Настройка на основе поз девушки
        min_angle = 130  # В нижней точке
        max_angle = 170  # В верхней точке

        # Конвертация угла в прогресс (100% при min_angle, 0% при max_angle)
        progress = np.clip((max_angle - angle) * (100 / (max_angle - min_angle)), 0, 100)
        return int(progress)

    progress_left, progress_right = 0, 0

    # Проверяем точки для правой ноги
    if (
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP] and
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE] and
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    ):
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

        progress_right = get_squat_progress(hip, knee, ankle)
        smoothed_progress_right = smooth_progress(progress_squat_buffer, progress_right)  # Буфер
        draw_vertical_progress_bar(frame, smoothed_progress_right, frame.shape[1] - 50, 50)  # Справа

    # Проверяем точки для левой ноги
    if (
        landmarks[mp_pose.PoseLandmark.LEFT_HIP] and
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE] and
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    ):
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

        progress_left = get_squat_progress(hip, knee, ankle)
        smoothed_progress_left = smooth_progress(progress_squat_buffer, progress_left)  # Буфер
        draw_vertical_progress_bar(frame, smoothed_progress_left, 30, 50)  # Слева


def track_bicep_curls(frame, landmarks):
    """Отслеживание бицепсовых сгибаний с боковыми прогресс-барами."""
    # Правая рука (справа)
    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER] and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW] and landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]:
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
        angle = calculate_angle(shoulder, elbow, wrist)

        progress = max(0, min(100, int(100 - (angle - 110) * (100 / (180 - 110)))))
        smoothed_progress = smooth_progress(progress_curl_right_buffer, progress)
        draw_vertical_progress_bar(frame, smoothed_progress, frame.shape[1] - 50, 50)  # Справа

    # Левая рука (слева)
    if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER] and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW] and landmarks[mp_pose.PoseLandmark.LEFT_WRIST]:
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        angle = calculate_angle(shoulder, elbow, wrist)

        progress = max(0, min(100, int(100 - (angle - 110) * (100 / (180 - 110)))))
        smoothed_progress = smooth_progress(progress_curl_left_buffer, progress)
        draw_vertical_progress_bar(frame, smoothed_progress, 10, 50)  # Слева

def track_lateral_raises(frame, landmarks):
    """Отслеживание махов гантелями (по бокам)."""
    # Правая рука (справа)
    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER] and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]:
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        angle = calculate_angle(hip, shoulder, elbow)

        progress = min(100, int((angle / 90) * 100))
        smoothed_progress = smooth_progress(progress_raise_right_buffer, progress)
        draw_vertical_progress_bar(frame, smoothed_progress, frame.shape[1] - 30, 50)  # Справа

    # Левая рука (слева)
    if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER] and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]:
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        angle = calculate_angle(hip, shoulder, elbow)

        progress = min(100, int((angle / 90) * 100))
        smoothed_progress = smooth_progress(progress_raise_left_buffer, progress)
        draw_vertical_progress_bar(frame, smoothed_progress, 10, 50)  # Слева

