
import cv2
import numpy as np
import math


IMAGE_PATH = "images/contoh2.jpg"


def to_grayscale(image):
    height, width = image.shape[:2]
    gray = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]
            value = 0.114 * int(b) + 0.587 * int(g) + 0.299 * int(r)
            gray[y, x] = int(value)

    return gray


def negative_transform(gray):
    height, width = gray.shape
    result = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            result[y, x] = 255 - int(gray[y, x])

    return result


def log_transform(gray):
    height, width = gray.shape
    result = np.zeros((height, width), dtype=np.uint8)
    c = 255.0 / math.log(256.0)

    for y in range(height):
        for x in range(width):
            value = int(gray[y, x])
            transformed = c * math.log(1 + value)

            if transformed > 255:
                transformed = 255

            result[y, x] = int(transformed)

    return result


def histogram_manual(gray):
    histogram = [0] * 256
    height, width = gray.shape

    for y in range(height):
        for x in range(width):
            value = int(gray[y, x])
            histogram[value] += 1

    return histogram


def equalize_histogram_manual(gray):
    height, width = gray.shape
    total_pixels = height * width

    histogram = histogram_manual(gray)

    cdf = [0] * 256
    cumulative = 0

    for i in range(256):
        cumulative += histogram[i]
        cdf[i] = cumulative

    cdf_min = 0

    for i in range(256):
        if cdf[i] > 0:
            cdf_min = cdf[i]
            break

    result = np.zeros((height, width), dtype=np.uint8)

    if total_pixels == cdf_min:
        return gray.copy()

    mapping = [0] * 256

    for i in range(256):
        value = (cdf[i] - cdf_min) / (total_pixels - cdf_min)
        value = value * 255

        if value < 0:
            value = 0
        if value > 255:
            value = 255

        mapping[i] = int(value)

    for y in range(height):
        for x in range(width):
            original = int(gray[y, x])
            result[y, x] = mapping[original]

    return result


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Gambar tidak ditemukan:", IMAGE_PATH)
        return

    gray = to_grayscale(image)
    negative = negative_transform(gray)
    logarithmic = log_transform(gray)
    equalized = equalize_histogram_manual(gray)

    cv2.imshow("Grayscale", gray)
    cv2.imshow("Transformasi Negatif", negative)
    cv2.imshow("Transformasi Logaritmik", logarithmic)
    cv2.imshow("Ekualisasi Histogram Manual", equalized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
