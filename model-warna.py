
import cv2
import numpy as np
import math


IMAGE_PATH = "images/contoh.jpg"


def rgb_to_cmyk(rgb):
    rgb_float = rgb.astype(np.float64) / 255.0

    r = rgb_float[:, :, 0]
    g = rgb_float[:, :, 1]
    b = rgb_float[:, :, 2]

    k = 1.0 - np.maximum(np.maximum(r, g), b)
    denominator = 1.0 - k

    c = np.zeros_like(k)
    m = np.zeros_like(k)
    y = np.zeros_like(k)

    valid = denominator > 0

    c[valid] = (1.0 - r[valid] - k[valid]) / denominator[valid]
    m[valid] = (1.0 - g[valid] - k[valid]) / denominator[valid]
    y[valid] = (1.0 - b[valid] - k[valid]) / denominator[valid]

    cmyk = np.stack([c, m, y, k], axis=2) * 255.0
    return np.clip(cmyk, 0, 255).astype(np.uint8)


def rgb_to_hsi(rgb):
    height, width = rgb.shape[:2]
    hsi = np.zeros((height, width, 3), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            r = float(rgb[y, x, 0]) / 255.0
            g = float(rgb[y, x, 1]) / 255.0
            b = float(rgb[y, x, 2]) / 255.0

            intensity = (r + g + b) / 3.0

            total = r + g + b
            saturation = 0.0 if total == 0 else (
                1.0 - 3.0 * min(r, g, b) / total
            )

            numerator = 0.5 * ((r - g) + (r - b))
            denominator = math.sqrt(
                (r - g) ** 2
                + (r - b) * (g - b)
            )

            if denominator == 0:
                hue = 0.0
            else:
                ratio = numerator / denominator
                ratio = max(-1.0, min(1.0, ratio))
                theta = math.degrees(math.acos(ratio))

                if b <= g:
                    hue = theta
                else:
                    hue = 360.0 - theta

            hsi[y, x] = [
                hue / 360.0 * 255.0,
                saturation * 255.0,
                intensity * 255.0
            ]

    return np.clip(hsi, 0, 255).astype(np.uint8)


def main():
    bgr = cv2.imread(IMAGE_PATH)

    if bgr is None:
        print("Gambar tidak ditemukan:", IMAGE_PATH)
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    cmyk = rgb_to_cmyk(rgb)
    hsi = rgb_to_hsi(rgb)

    # OpenCV menampilkan gambar dalam urutan BGR.
    cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Setiap kanal ditampilkan sebagai citra grayscale.
    cv2.imshow("HSV - Hue", hsv[:, :, 0])
    cv2.imshow("HSV - Saturation", hsv[:, :, 1])
    cv2.imshow("HSV - Value", hsv[:, :, 2])

    cv2.imshow("CMYK - Cyan", cmyk[:, :, 0])
    cv2.imshow("CMYK - Magenta", cmyk[:, :, 1])
    cv2.imshow("CMYK - Yellow", cmyk[:, :, 2])
    cv2.imshow("CMYK - Black", cmyk[:, :, 3])

    cv2.imshow("HSI - Hue", hsi[:, :, 0])
    cv2.imshow("HSI - Saturation", hsi[:, :, 1])
    cv2.imshow("HSI - Intensity", hsi[:, :, 2])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
