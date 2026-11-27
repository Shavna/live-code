
import cv2
import numpy as np

IMAGE_PATH = "images/contoh.jpg"


def read_image(path):
    image = cv2.imread(path)

    if image is None:
        print("Gambar tidak ditemukan:", path)
        return None

    return image


def show_image(title, image):
    cv2.imshow(title, image)


def filter_color_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 100, 100])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([160, 100, 100])
    upper_red_2 = np.array([179, 255, 255])

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    mask = cv2.bitwise_or(mask_1, mask_2)
    result = cv2.bitwise_and(image, image, mask=mask)

    return mask, result


def filter_color_video():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Webcam tidak dapat dibuka.")
        return

    while True:
        ret, frame = camera.read()

        if not ret:
            break

        mask, result = filter_color_image(frame)

        cv2.imshow("Video Asli", frame)
        cv2.imshow("Mask Merah", mask)
        cv2.imshow("Hasil Filter Video", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def main():
    image = read_image(IMAGE_PATH)

    if image is not None:
        mask, result = filter_color_image(image)

        show_image("Citra Asli", image)
        show_image("Mask Merah", mask)
        show_image("Hasil Filter Citra", result)

        print("Tekan sembarang tombol pada jendela gambar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    choice = input("Uji filter video webcam? (y/n): ").lower()

    if choice == "y":
        filter_color_video()


if __name__ == "__main__":
    main()
