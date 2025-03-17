import cv2 as cv
import numpy as np
import argparse
import math


# Функция для вычисления угла между 3 точками
def calculate_angle(a, b, c):
    # Получаем координаты точек
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Вычисляем векторы
    ab = a - b
    bc = c - b

    # Вычисляем угол между векторами
    cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))

    # Преобразуем угол в градусы
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180.0 / np.pi
    return angle