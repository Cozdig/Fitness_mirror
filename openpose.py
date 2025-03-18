import cv2
import mediapipe as mp
import math
import numpy as np
import tkinter as tk
from excercise import calculate_angle, track_squats, track_bicep_curls, track_lateral_raises

# Инициализация MediaPipe для отслеживания позы
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Переменная для отслеживания выбранного упражнения
selected_exercise = None
progress = 0  # Изначально прогресс равен 0


# Функция для выбора упражнения через GUI
def set_exercise(choice):
    global selected_exercise
    if choice == "1":
        selected_exercise = track_bicep_curls
    elif choice == "2":
        selected_exercise = track_squats
    elif choice == "3":
        selected_exercise = track_lateral_raises
    else:
        selected_exercise = None

    # Закрыть GUI после выбора
    root.quit()

# Создание графического интерфейса для выбора упражнения
def create_gui():
    global root
    root = tk.Tk()
    root.title("Выберите упражнение")

    label = tk.Label(root, text="Выберите упражнение:", font=("Arial", 14))
    label.pack(pady=20)

    button_bicep_curl = tk.Button(root, text="Бицепсовые сгибания", width=30, height=2, command=lambda: set_exercise("1"))
    button_bicep_curl.pack(pady=10)

    button_squat = tk.Button(root, text="Приседания", width=30, height=2, command=lambda: set_exercise("2"))
    button_squat.pack(pady=10)

    button_dumbbell_fly = tk.Button(root, text="Махи гантелями", width=30, height=2, command=lambda: set_exercise("3"))
    button_dumbbell_fly.pack(pady=10)

    button_exit = tk.Button(root, text="Выход", width=30, height=2, command=root.quit)
    button_exit.pack(pady=10)

    root.mainloop()

# Запуск GUI для выбора упражнения
create_gui()

# Открытие видеопотока
cap = cv2.VideoCapture(0)

while selected_exercise:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование изображения для MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Вызов функции отслеживания выбранного упражнения
        selected_exercise(frame, landmarks)

        # Отображение ключевых точек
        for landmark in landmarks:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Показать кадр
    cv2.imshow("Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
