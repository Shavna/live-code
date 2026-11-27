
import cv2
import numpy as np


IMAGE_PATH = "images/contoh3.jpg"


def mean_filter_manual(gray, kernel_size=3):
    height, width = gray.shape
    result = np.zeros((height, width), dtype=np.uint8)
    pad = kernel_size // 2

    padded = np.pad(
        gray.astype(np.float64),
        ((pad, pad), (pad, pad)),
        mode="edge"
    )

    for y in range(height):
        for x in range(width):
            total = 0.0

            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    total += padded[y + ky, x + kx]

            result[y, x] = int(total / (kernel_size * kernel_size))

    return result


def median_filter_manual(gray, kernel_size=3):
    height, width = gray.shape
    result = np.zeros((height, width), dtype=np.uint8)
    pad = kernel_size // 2

    padded = np.pad(
        gray,
        ((pad, pad), (pad, pad)),
        mode="edge"
    )

    for y in range(height):
        for x in range(width):
            values = []

            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    values.append(
                        int(padded[y + ky, x + kx])
                    )

            values.sort()
            result[y, x] = values[len(values) // 2]

    return result


def laplacian_filter_manual(gray):
    height, width = gray.shape
    result = np.zeros((height, width), dtype=np.uint8)

    kernel = [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ]

    padded = np.pad(
        gray.astype(np.float64),
        ((1, 1), (1, 1)),
        mode="edge"
    )

    for y in range(height):
        for x in range(width):
            total = 0.0

            for ky in range(3):
                for kx in range(3):
                    total += (
                        padded[y + ky, x + kx]
                        * kernel[ky][kx]
                    )

            value = abs(total)

            if value > 255:
                value = 255

            result[y, x] = int(value)

    return result


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Gambar tidak ditemukan:", IMAGE_PATH)
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_result = mean_filter_manual(gray)
    median_result = median_filter_manual(gray)
    laplacian_result = laplacian_filter_manual(gray)

    cv2.imshow("Citra Asli", gray)
    cv2.imshow("Mean Filter Manual", mean_result)
    cv2.imshow("Median Filter Manual", median_result)
    cv2.imshow("Laplacian Manual", laplacian_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
