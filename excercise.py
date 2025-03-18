import cv2 as cv
import numpy as np
import argparse
import math


# Функция для вычисления угла между 3 точками
import numpy as np

def calculate_angle(a, b, c):
    # Вектор из точки a в точку b
    ab = np.array(a) - np.array(b)
    # Вектор из точки b в точку c
    bc = np.array(c) - np.array(b)

    # Вычисление нормы векторов
    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)

    # Если один из векторов имеет нулевую норму, угол не может быть вычислен
    if norm_ab == 0 or norm_bc == 0:
        return 0  # Возвращаем угол 0 или любое другое значение по вашему усмотрению

    # Вычисление косинуса угла между векторами
    cos_angle = np.dot(ab, bc) / (norm_ab * norm_bc)

    # Ограничиваем значение косинуса угла, чтобы избежать ошибок из-за погрешностей в вычислениях
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Вычисление угла в радианах и преобразование в градусы
    angle = np.arccos(cos_angle)
    angle = np.degrees(angle)

    return angle