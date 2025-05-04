import cv2
import numpy as np
import math


def preprocess_frame(frame, blur_kernel_size=3):
    """
    HSV uzayına çevirmeden önce gürltüyü azaltmak için bulanıklaştırma uyguluyoruz.
    Args:
        frame (np.ndarray): BGR görüntü.
        blur_kernel_size (int): Bulanıklaştırma katsayısı (tek sayı olmalı).

    Returns:
        np.ndarray: bulanıklaştırılmış görüntü.
    """
    return cv2.GaussianBlur(frame, (blur_kernel_size, blur_kernel_size), 0)


def get_dynamic_kernel(img_area):
    """
    Görüntünün boyutuna göre kernel belirleme.

    Args:
        img_area (int): Görüntünün piksel alanı (Yükseklik * Genişlik).

    Returns:
        np.ndarray:  morphological işlemler için kernel arrayi.
    """
    kernel_size = max(3, int(math.sqrt(img_area) // 100))
    if kernel_size % 2 == 0:  # kerneli tek sayı yapma
        kernel_size += 1
    return np.ones((kernel_size, kernel_size), np.uint8)


def apply_morphological_filters(mask, kernel, open_iterations=1, erode_iterations=1, dilate_iterations=1, apply_open=True):
    """
    morphologic işlemleri yapar ve mask ı temizler.

    Args:
        mask (np.ndarray): Binary mask .
        kernel (np.ndarray):  morphological işlemler için kernel katsayısı.
        open_iterations (int): morphological açılış için iterasyon sayısı.
        erode_iterations (int): erosion için iterasyon katsayısı.
        dilate_iterations (int):  dilation için iterasyon katsayısı.

    Returns:
        np.ndarray: filtrelenmiş maske.
    """
    if apply_open:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    mask = cv2.erode(mask, kernel, iterations=erode_iterations)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
    return mask


def filter_contour(cnt, area_threshold, min_circularity=0.4):
    """
    Alan ve daireselliğe göre kontur filtreleme.

    Args:
        cnt (np.ndarray): filtrelencek kontur.
        area_threshold (float): Minimum kontur alanı.
        min_circularity (float): Minimum dairesellik eşik alanı.

    Returns:
        bool:  kontur filtreyi geçerse True, Aksi takdirde False.
    """
    area = cv2.contourArea(cnt)
    if area < area_threshold:
        return False

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False

    circularity = 4 * math.pi * area / (perimeter * perimeter)
    return circularity >= min_circularity