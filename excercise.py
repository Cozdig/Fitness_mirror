from datetime import time
import threading
from playsound import playsound
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque

def play_beep():
    threading.Thread(target=playsound, args=("beep.mp3",), daemon=True).start()

# Фильтр для сглаживания прогресса (чтобы полоска не прыгала)
class SmoothingFilter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return int(self.value)


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

# Счетчики повторений
squat_counter = 0
curl_counter = 0
raise_counter = 0

# Флаги выполнения упражнения обеими конечностями
squat_completed = False
curl_completed = False
raise_completed = False

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

def draw_counter(frame, count, x, y):
    """Рисует счетчик повторений."""
    text = f"Reps: {count}"
    cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)  # Черная обводка
    cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)  # Белый текст

def track_squats(frame, landmarks):
    """Отслеживание фронтальных приседаний с учетом обеих ног."""
    global squat_counter, squat_completed

    def get_squat_progress(hip, knee, ankle):
        """Вычисляет прогресс приседаний на основе угла бедра относительно вертикали."""
        angle = calculate_angle(hip, knee, ankle)
        min_angle, max_angle = 140, 170
        return np.clip((max_angle - angle) * (100 / (max_angle - min_angle)), 0, 100)

    progress_left, progress_right = 0, 0

    # Проверяем правую ногу
    if all(landmarks[pt] for pt in [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]):
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
        progress_right = smooth_progress(progress_squat_buffer, get_squat_progress(hip, knee, ankle))

    # Проверяем левую ногу
    if all(landmarks[pt] for pt in [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE]):
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
        progress_left = smooth_progress(progress_squat_buffer, get_squat_progress(hip, knee, ankle))

    # Проверяем завершение упражнения обеими ногами
    if progress_left >= 95 and progress_right >= 95:
        squat_completed = True
    elif squat_completed and progress_left <= 15 and progress_right <= 15:
        squat_counter += 1
        squat_completed = False
        play_beep()

    # Рисуем прогресс-бары
    draw_vertical_progress_bar(frame, progress_left, frame.shape[1] - 50, 50)  # Слева
    draw_vertical_progress_bar(frame, progress_right, 30, 50)  # Справа
    draw_counter(frame, squat_counter, frame.shape[1] // 2 - 50, 50)


# Обновляем пределы углов для бицепсовых сгибаний
MIN_ANGLE = 70  # 100% прогресса
MAX_ANGLE = 160  # 0% прогресса

def track_bicep_curls(frame, landmarks):
    """Отслеживание бицепсовых сгибаний с плавным прогресс-баром."""
    global curl_counter, curl_completed

    progress_left, progress_right = 0, 0

    # Правая рука
    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER] and \
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW] and \
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]:
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

        angle = calculate_angle(shoulder, elbow, wrist)

        # Обновленный прогресс: 100% при 70°, 0% при 160°
        progress_right = np.clip(int(100 - (angle - MIN_ANGLE) * (100 / (MAX_ANGLE - MIN_ANGLE))), 0, 100)
        smoothed_progress_right = smooth_progress(progress_curl_right_buffer, progress_right)
        draw_vertical_progress_bar(frame, smoothed_progress_right, 30, 50)  # Справа

    # Левая рука
    if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER] and \
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW] and \
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST]:
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

        angle = calculate_angle(shoulder, elbow, wrist)

        # Обновленный прогресс
        progress_left = np.clip(int(100 - (angle - MIN_ANGLE) * (100 / (MAX_ANGLE - MIN_ANGLE))), 0, 100)
        smoothed_progress_left = smooth_progress(progress_curl_left_buffer, progress_left)
        draw_vertical_progress_bar(frame, smoothed_progress_left, frame.shape[1] - 50, 50)  # Слева

    # Логика засчёта повтора
    if progress_left >= 85 and progress_right >= 85:
        curl_completed = True

    if curl_completed and progress_left <= 15 and progress_right <= 15:
        curl_counter += 1
        curl_completed = False
        play_beep()  # Звуковой сигнал

    # Отображаем счётчик по центру сверху
    draw_counter(frame, curl_counter, frame.shape[1] // 2 - 50, 50)

def track_lateral_raises(frame, landmarks):
    """Отслеживание махов гантелями (lateral raises)."""
    global raise_counter, raise_completed

    def get_raise_progress(hip, shoulder, wrist):
        """Вычисляет прогресс махов на основе угла руки относительно корпуса."""
        angle = calculate_angle(hip, shoulder, wrist)  # Угол между бедром, плечом и запястьем
        return min(100, max(0, int((angle - 10) * (100 / (90 - 10)))))  # 10° - опущено, 90° - уровень плеч

    progress_right, progress_left = 0, 0

    # Правая рука
    if all(landmarks[pt] for pt in [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_WRIST]):
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
        progress_right = smooth_progress(progress_raise_right_buffer, get_raise_progress(hip, shoulder, wrist))

    # Левая рука
    if all(landmarks[pt] for pt in [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_WRIST]):
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        progress_left = smooth_progress(progress_raise_left_buffer, get_raise_progress(hip, shoulder, wrist))

    # Проверяем завершение повторения (обе руки подняты)
    if progress_left >= 95 and progress_right >= 95:
        raise_completed = True
    elif raise_completed and progress_left <= 15 and progress_right <= 15:  # Теперь 15%, а не 10%
        raise_counter += 1
        raise_completed = False
        play_beep()

    # Рисуем прогресс-бары
    draw_vertical_progress_bar(frame, progress_left, frame.shape[1] - 50, 50)  # Слева
    draw_vertical_progress_bar(frame, progress_right, 30, 50)  # Справа

    # Рисуем счетчик повторов
    draw_counter(frame, raise_counter, frame.shape[1] // 2 - 50, 50)