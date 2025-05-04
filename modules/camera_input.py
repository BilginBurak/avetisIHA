import cv2


def get_camera_stream(camera_index=0, width=640, height=480):
    """
    Kamera akışını başlatır ve her kareyi yield eder.

    Args:
        camera_index (int): Kamera indeksi. Genellikle USB için 0, CSI kameralar için 1 veya 2.
        width (int): Genişlik çözünürlüğü.
        height (int): Yükseklik çözünürlüğü.

    Yields:
        np.ndarray: Kamera karesi.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise IOError("Kamera başlatılamadı!")

    # Çözünürlük ayarlanıyor
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kamera görüntüsü alınamadı!")
                break
            yield frame
    finally:
        cap.release()
        cv2.destroyAllWindows()

