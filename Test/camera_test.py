import cv2

cap = cv2.VideoCapture(0)  # Gerekirse 1 veya 2 yap, ya da CSI için özel pipeline
if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı!")
        break

    cv2.imshow("Kamera Testi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
