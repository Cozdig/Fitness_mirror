from datetime import time

import cv2 as cv
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from collections import deque
import os
import sys
from PIL import Image, ImageDraw, ImageFont
from excercise import track_lateral_raises, track_squats, track_bicep_curls

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
        """Открытие камеры"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

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


def set_exercise(choice):
    """Функция выбора упражнения"""
    global selected_exercise, exit_to_menu, root
    if root:
        root.destroy()

    camera.close_camera()  # Закрываем камеру при возврате в меню
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
        frame = draw_text(frame, "Нажмите M для меню", (20, frame.shape[0] - 40))

        cv.imshow("Pose", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('m'):
            exit_to_menu = True
            selected_exercise = None
            break

camera.close_camera()
