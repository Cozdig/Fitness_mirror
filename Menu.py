from tkinter import ttk
import time
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import threading
from playsound import playsound
import cv2 as cv
import numpy as np
import mediapipe as mp
import pygame
from collections import deque
import tkinter as tk
from tkinter import simpledialog

if getattr(sys, 'frozen', False):
    BASE_PATH = sys._MEIPASS  # PyInstaller хранит файлы здесь
else:
    BASE_PATH = os.path.abspath(".")

# Пути к файлам
MUSIC_PATH = os.path.join(BASE_PATH, "music", "background.mp3")
BEEP_PATH = os.path.join(BASE_PATH, "beep.mp3")
GOAL_SOUND_PATH = os.path.join(BASE_PATH, "goal_reached.mp3")
VIDEO_PATH_TEMPLATE = os.path.join(BASE_PATH, "videos", "{}.mp4")

print("Ищем музыку в:", MUSIC_PATH)

music_playing = False

def toggle_music():
    """Переключает фоновую музыку (включает/выключает)."""
    global music_playing
    if music_playing:
        pygame.mixer.music.pause()
    else:
        pygame.mixer.music.unpause()
    music_playing = not music_playing

def play_music():
    """Запускает фоновую музыку."""
    pygame.mixer.init()
    pygame.mixer.music.set_volume(0.3)
    pygame.mixer.music.load(MUSIC_PATH)  # Укажи путь к файлу с музыкой
    pygame.mixer.music.play(-1)  # -1 означает бесконечное повторение

def stop_music():
    """Останавливает фоновую музыку."""
    pygame.mixer.music.stop()

def play_beep():
    threading.Thread(target=playsound, args=(BEEP_PATH,), daemon=True).start()


# Функция для выбора цели по повторениям
root = tk.Tk()
root.withdraw()  # Скрываем главное окно
squat_goal = simpledialog.askinteger("Цель", "Введите количество повторений для приседаний", minvalue=1, maxvalue=100)
curl_goal = simpledialog.askinteger("Цель", "Введите количество повторений для бицепса", minvalue=1, maxvalue=100)
raise_goal = simpledialog.askinteger("Цель", "Введите количество повторений для махов", minvalue=1, maxvalue=100)
root.destroy()

# Функция для звукового сигнала при достижении цели
def play_goal_reached():
    threading.Thread(target=playsound, args=(GOAL_SOUND_PATH,), daemon=True).start()


def draw_goal_counter(frame, count, goal, x, y):
    """Рисует счетчик повторений с целью."""
    text = f"Reps: {count}/{goal}"
    cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)  # Черная обводка
    cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)  # Белый текст

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

# Создаём фильтры для сглаживания прогресс-бара
smooth_left = SmoothingFilter(alpha=0.5)
smooth_right = SmoothingFilter(alpha=0.5)

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

def reset_counters():
    global squat_counter, curl_counter, raise_counter
    squat_counter = 0
    curl_counter = 0
    raise_counter = 0

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
    global squat_counter, squat_completed, squat_goal

    if squat_counter >= squat_goal:
        draw_goal_counter(frame, squat_counter, squat_goal, frame.shape[1] // 2 - 50, 50)
        return

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
        if squat_counter == squat_goal:
            play_goal_reached()
        else:
            play_beep()

    # Рисуем прогресс-бары
    draw_vertical_progress_bar(frame, progress_left, frame.shape[1] - 50, 50)  # Слева
    draw_vertical_progress_bar(frame, progress_right, 30, 50)  # Справа
    draw_counter(frame, squat_counter, frame.shape[1] // 2 - 50, 50)
    draw_goal_counter(frame, squat_counter, squat_goal, frame.shape[1] // 2 - 50, 50)

# Обновляем пределы углов для бицепсовых сгибаний
MIN_ANGLE = 70  # 100% прогресса
MAX_ANGLE = 160  # 0% прогресса

def track_bicep_curls(frame, landmarks):
    """Отслеживание бицепсовых сгибаний с плавным прогресс-баром."""
    global curl_counter, curl_completed, curl_goal
    if curl_counter >= curl_goal:
        draw_goal_counter(frame, curl_counter, curl_goal, frame.shape[1] // 2 - 50, 50)
        return
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
        smoothed_progress_right = smooth_right.update(progress_right)
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
        smoothed_progress_left = smooth_left.update(progress_left)
        draw_vertical_progress_bar(frame, smoothed_progress_left, frame.shape[1] - 50, 50)  # Слева

    # Логика засчёта повтора
    if progress_left >= 85 and progress_right >= 85:
        curl_completed = True

    if curl_completed and progress_left <= 15 and progress_right <= 15:
        curl_counter += 1
        curl_completed = False
        if curl_counter == curl_goal:
            play_goal_reached()
        else:
            play_beep() # Звуковой сигнал

    # Отображаем счётчик по центру сверху
    draw_counter(frame, curl_counter, frame.shape[1] // 2 - 50, 50)
    draw_goal_counter(frame, curl_counter, curl_goal, frame.shape[1] // 2 - 50, 50)


def track_lateral_raises(frame, landmarks):
    """Отслеживание махов гантелями (lateral raises)."""
    global raise_counter, raise_completed, raise_goal
    if raise_counter >= raise_goal:
        draw_goal_counter(frame, raise_counter, raise_goal, frame.shape[1] // 2 - 50, 50)
        return

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
        if raise_counter == raise_goal:
            play_goal_reached()
        else:
            play_beep()

    # Рисуем прогресс-бары
    draw_vertical_progress_bar(frame, progress_left, frame.shape[1] - 50, 50)  # Слева
    draw_vertical_progress_bar(frame, progress_right, 30, 50)  # Справа

    # Рисуем счетчик повторов
    draw_counter(frame, raise_counter, frame.shape[1] // 2 - 50, 50)
    draw_goal_counter(frame, raise_counter, raise_goal, frame.shape[1] // 2 - 50, 50)




# Инициализация MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Буферы для сглаживания данных
N = 5
progress_buffers = {
    "squat": deque(maxlen=N),
    "curl_right": deque(maxlen=N),
    "curl_left": deque(maxlen=N),
    "raise_right": deque(maxlen=N),
    "raise_left": deque(maxlen=N),
}

# Переменные состояния
selected_exercise = None
exit_to_menu = False
root = None

# Словарь выбора упражнений
EXERCISES = {
    "1": track_bicep_curls,
    "2": track_squats,
    "3": track_lateral_raises,
}

def draw_text(frame, text, position, font_path="arial.ttf", font_size=20, color=(255, 255, 255), shadow_color=(0, 0, 0)):
    """ Отображает читаемый текст с тенью для лучшей видимости """
    pil_image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Тень для текста
    shadow_offset = 1
    draw.text((position[0] + shadow_offset, position[1] + shadow_offset), text, font=font, fill=shadow_color)

    # Основной текст
    draw.text(position, text, font=font, fill=color)

    return cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)



class CameraHandler:
    """Класс для работы с камерой"""

    def __init__(self, width=640, height=480):
        self.cap = None
        self.width = width
        self.height = height

    def open_camera(self):
        """Открытие камеры с задержкой"""
        if self.cap:
            self.close_camera()

        time.sleep(1)  # Даем системе немного времени

        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.cap.isOpened():
            print("✅ Камера успешно открыта")
        else:
            print("❌ Ошибка: камера не открывается")
    def close_camera(self):
        """Закрытие камеры"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv.destroyAllWindows()

    def read_frame(self):
        """Чтение кадра с камеры"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return ret, frame
        return False, None


camera = CameraHandler()


def play_video(video_path):
    """Воспроизводит видео-пример перед началом упражнения с возможностью паузы."""
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}")
        return

    paused = False  # Флаг паузы

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            cv.imshow("Видео-пример", frame)

        key = cv.waitKey(30) & 0xFF

        if key == ord('q'):  # Закрытие окна по 'q'
            break
        elif key == ord(' '):  # Пауза/возобновление по пробелу
            paused = not paused

    cap.release()
    cv.destroyWindow("Видео-пример")


def set_exercise(choice):
    """Функция выбора упражнения"""
    global selected_exercise, exit_to_menu, root
    if root:
        root.destroy()

    camera.close_camera()  # Закрываем камеру при возврате в меню
    video_path = VIDEO_PATH_TEMPLATE.format(choice)  # Передаем путь правильно
    play_video(video_path)
    if os.path.exists(video_path):
        play_video(video_path)
    reset_counters()
    play_music()
    selected_exercise = EXERCISES.get(choice, None)
    exit_to_menu = False


def create_gui():
    """Создание графического интерфейса"""
    global root
    if root:
        try:
            if root.winfo_exists():
                return
        except tk.TclError:
            root = None  # Если окно уничтожено, сбрасываем root

    # Закрываем камеру перед возвратом в меню
    camera.close_camera()

    root = tk.Tk()
    root.title("Выберите упражнение")
    root.geometry("450x400")
    root.configure(bg="#1E1E1E")

    label = ttk.Label(root, text="Выберите упражнение:", font=("Arial", 16, "bold"), background="#1E1E1E",
                      foreground="white")
    label.pack(pady=20)

    buttons = [
        ("💪 Упражнение на бицепс", "1"),
        ("⭐ Приседания", "2"),
        ("🏋 Махи гантелями", "3"),
        ("❌ Выход", "exit"),
    ]

    for text, value in buttons:
        btn = tk.Button(root, text=text,
                        command=(lambda v=value: exit_application() if v == "exit" else set_exercise(v)),
                        bg="white", font=("Arial", 12), relief="flat", activebackground="#cccccc")
        btn.pack(pady=10, padx=20, fill="x")

        # Анимация кнопок (изменение цвета при наведении)
        btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#cccccc"))
        btn.bind("<Leave>", lambda e, b=btn: b.config(bg="white"))

    root.protocol("WM_DELETE_WINDOW", exit_application)
    root.mainloop()


def exit_application():
    """Закрытие программы корректно"""
    camera.close_camera()
    sys.exit()


while True:
    create_gui()
    if selected_exercise is None:
        break

    camera.open_camera()  # Открываем камеру перед началом упражнения

    while selected_exercise and not exit_to_menu:
        ret, frame = camera.read_frame()
        if not ret:
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            selected_exercise(frame, landmarks)

        # Добавляем текст в кадр
        frame = draw_text(frame, "Нажмите M для меню | Нажмите T для остановки музыки", (20, frame.shape[0] - 40))

        cv.imshow("Pose", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('m'):
            stop_music()
            exit_to_menu = True
            selected_exercise = None
            break
        elif key == ord('t'):  # Переключение музыки
            toggle_music()
camera.close_camera()
