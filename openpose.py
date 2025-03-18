import cv2 as cv
import numpy as np
import argparse
import tkinter as tk
from excercise import calculate_angle

# Парсинг аргументов
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
args = parser.parse_args()

# Инициализация BODY_PARTS и POSE_PAIRS
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

inWidth = args.width
inHeight = args.height

# Загрузка сети
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# Переменная для отслеживания выбранного упражнения
selected_exercise = None
progress = 0  # Изначально прогресс равен 0

# Функция для отслеживания бицепсовых сгибаний
def track_bicep_curls(frame, points):
    global progress

    if points[BODY_PARTS["RShoulder"]] and points[BODY_PARTS["RElbow"]] and points[BODY_PARTS["RWrist"]]:
        shoulder_r = points[BODY_PARTS["RShoulder"]]
        elbow_r = points[BODY_PARTS["RElbow"]]
        wrist_r = points[BODY_PARTS["RWrist"]]

        angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)

        # Прогресс выполнения бицепса (чем ближе угол к 180°, тем больше прогресс)
        progress = min((angle_r / 180) * 100, 100)
        cv.putText(frame, f'Bicep curl progress (right): {int(progress)}%', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if points[BODY_PARTS["LShoulder"]] and points[BODY_PARTS["LElbow"]] and points[BODY_PARTS["LWrist"]]:
        shoulder_l = points[BODY_PARTS["LShoulder"]]
        elbow_l = points[BODY_PARTS["LElbow"]]
        wrist_l = points[BODY_PARTS["LWrist"]]

        angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)

        # Прогресс выполнения бицепса для левой руки
        progress = min((angle_l / 180) * 100, 100)
        cv.putText(frame, f'Bicep curl progress (left): {int(progress)}%', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Функция для отслеживания приседаний
def track_squats(frame, points):
    global progress

    if points[BODY_PARTS["RHip"]] and points[BODY_PARTS["RKnee"]] and points[BODY_PARTS["RAnkle"]]:
        hip_r = points[BODY_PARTS["RHip"]]
        knee_r = points[BODY_PARTS["RKnee"]]
        ankle_r = points[BODY_PARTS["RAnkle"]]

        angle_r = calculate_angle(hip_r, knee_r, ankle_r)

        # Прогресс выполнения приседания (чем ниже угол между бедром и голенью, тем больше прогресс)
        if angle_r < 100:  # Если угол между бедром и голенью меньше 100, то приседания выполнены правильно
            progress = min((1 - (angle_r / 100)) * 100, 100)
        else:
            progress = 0  # Если угол больше 100, прогресс равен 0 (не присел)
        cv.putText(frame, f'Squat progress (right): {int(progress)}%', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if points[BODY_PARTS["LHip"]] and points[BODY_PARTS["LKnee"]] and points[BODY_PARTS["LAnkle"]]:
        hip_l = points[BODY_PARTS["LHip"]]
        knee_l = points[BODY_PARTS["LKnee"]]
        ankle_l = points[BODY_PARTS["LAnkle"]]

        angle_l = calculate_angle(hip_l, knee_l, ankle_l)

        # Прогресс выполнения приседания для левой ноги
        if angle_l < 100:
            progress = min((1 - (angle_l / 100)) * 100, 100)
        else:
            progress = 0  # Если угол больше 100, прогресс равен 0
        cv.putText(frame, f'Squat progress (left): {int(progress)}%', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Функция для отслеживания махов гантелями
def track_dumbbell_fly(frame, points):
    global progress

    if points[BODY_PARTS["RShoulder"]] and points[BODY_PARTS["RElbow"]] and points[BODY_PARTS["RWrist"]]:
        shoulder_r = points[BODY_PARTS["RShoulder"]]
        elbow_r = points[BODY_PARTS["RElbow"]]
        wrist_r = points[BODY_PARTS["RWrist"]]

        angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)

        # Прогресс выполнения махов гантелями (чем больше угол, тем выше прогресс)
        if angle_r > 160:  # Угол должен быть как минимум 160° для правильного выполнения
            progress = min((angle_r / 180) * 100, 100)
            cv.putText(frame, f'Dumbbell fly progress (right): {int(progress)}%', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            progress = 0  # Если угол меньше 160°, прогресс равен 0
            cv.putText(frame, f'Dumbbell fly progress (right): {int(progress)}%', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if points[BODY_PARTS["LShoulder"]] and points[BODY_PARTS["LElbow"]] and points[BODY_PARTS["LWrist"]]:
        shoulder_l = points[BODY_PARTS["LShoulder"]]
        elbow_l = points[BODY_PARTS["LElbow"]]
        wrist_l = points[BODY_PARTS["LWrist"]]

        angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)

        # Прогресс выполнения махов гантелями для левой руки
        if angle_l > 160:
            progress = min((angle_l / 180) * 100, 100)
            cv.putText(frame, f'Dumbbell fly progress (left): {int(progress)}%', (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            progress = 0
            cv.putText(frame, f'Dumbbell fly progress (left): {int(progress)}%', (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Функция для выбора упражнения через GUI
def set_exercise(choice):
    global selected_exercise
    if choice == "1":
        selected_exercise = track_bicep_curls
    elif choice == "2":
        selected_exercise = track_squats
    elif choice == "3":
        selected_exercise = track_dumbbell_fly
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

# После выбора упражнения открываем камеру и начинаем отслеживание
cap = cv.VideoCapture(args.input if args.input else 0)

while selected_exercise:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    # Ввод в сеть
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # Оставляем только 19 частей тела

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        if conf > args.thr:
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Вызов функции отслеживания выбранного упражнения
    selected_exercise(frame, points)

    # Соединение точек в POSE_PAIRS
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    # Отображение производительности сети
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Показать кадр
    cv.imshow("pose", frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
