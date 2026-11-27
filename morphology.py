
import cv2
import numpy as np


IMAGE_PATH = "images/contoh.jpg"


def threshold_manual(gray, threshold=127):
    height, width = gray.shape
    binary = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if gray[y, x] >= threshold:
                binary[y, x] = 255

    return binary


def morphology_manual(binary, operation, kernel_size=3):
    height, width = binary.shape
    result = np.zeros((height, width), dtype=np.uint8)
    pad = kernel_size // 2

    padded = np.pad(
        binary,
        ((pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0
    )

    for y in range(height):
        for x in range(width):
            values = []

            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    values.append(
                        int(padded[y + ky, x + kx])
                    )

            if operation == "erode":
                result[y, x] = (
                    255 if all(v == 255 for v in values) else 0
                )

            elif operation == "dilate":
                result[y, x] = (
                    255 if any(v == 255 for v in values) else 0
                )

    return result


def opening(binary, kernel_size=3):
    eroded = morphology_manual(
        binary, "erode", kernel_size
    )

    return morphology_manual(
        eroded, "dilate", kernel_size
    )


def closing(binary, kernel_size=3):
    dilated = morphology_manual(
        binary, "dilate", kernel_size
    )

    return morphology_manual(
        dilated, "erode", kernel_size
    )


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Gambar tidak ditemukan:", IMAGE_PATH)
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = threshold_manual(gray)

    eroded = morphology_manual(binary, "erode")
    dilated = morphology_manual(binary, "dilate")
    opened = opening(binary)
    closed = closing(binary)

    cv2.imshow("Citra Biner", binary)
    cv2.imshow("Erosi", eroded)
    cv2.imshow("Dilasi", dilated)
    cv2.imshow("Opening", opened)
    cv2.imshow("Closing", closed)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
