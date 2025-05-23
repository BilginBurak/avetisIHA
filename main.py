from modules.camera_input import get_camera_stream
from modules.vision_engine import process_frame
import cv2

def main():
    rtsp_url = "rtsp://192.168.144.25:8554/main.264"
    for frame in get_camera_stream(rtsp_url):
        result = process_frame(frame, area_threshold=500, debug=True)

        for detection in result["detections"]:
            print(f"Detected {detection['color']} {detection['shape']} at {detection['center']}")

        # Görüntüyü göster
        if result["annotated_frame"] is not None:
            cv2.imshow("Detection", result["annotated_frame"])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
main()
