from modules.camera_input import get_camera_frame
from modules.vision_engine import process_frame
import cv2

def main_loop():
    while True:
        frame = get_camera_frame()
        if frame is None:
            continue

        result = process_frame(frame)

        if result["detected"]:
            x, y = result["center"]
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"Area: {result['area']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Görüntü", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
