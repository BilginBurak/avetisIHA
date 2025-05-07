import cv2
import socket
import struct

def get_camera_stream(rtsp_url, udp_ip, udp_port, packet_size=1024):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise IOError("RTSP akışı başlatılamadı!")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Görüntü alınamadı!")
                break

            # --- UDP üzerinden gönder ---
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            result, encoded_img = cv2.imencode('.jpg', frame, encode_param)

            if not result:
                print("Kodlama hatası")
                continue

            data = encoded_img.tobytes()
            total_packets = (len(data) + packet_size - 1) // packet_size

            for i in range(total_packets):
                start = i * packet_size
                end = start + packet_size
                chunk = data[start:end]

                # Başlık: frame_id, packet_id, total_packets
                header = struct.pack('!HHH', frame_id, i, total_packets)
                sock.sendto(header + chunk, (udp_ip, udp_port))

            frame_id = (frame_id + 1) % 65536

            # --- Kareyi dışa aktar ---
            yield frame

    finally:
        cap.release()
        sock.close()
        cv2.destroyAllWindows()
