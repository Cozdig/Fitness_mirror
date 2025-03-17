import cv2

# Инициализируем захват с камеры (обычно камера с индексом 0, если их несколько, индекс можно изменить)
cap = cv2.VideoCapture(0)

# Проверяем, открылся ли видеопоток
if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру.")
    exit()

# Захват кадров в реальном времени
while True:
    # Считываем кадр
    ret, frame = cap.read()

    # Если кадр не был считан, завершаем цикл
    if not ret:
        print("Не удалось получить кадр.")
        break

    # Отображаем кадр
    cv2.imshow("Video Stream", frame)

    # Прерываем цикл, если нажата клавиша 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы после завершения
cap.release()
cv2.destroyAllWindows()