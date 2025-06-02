import socket
import time
from modules.vision_engine import process_frame
from modules.camera_input import get_camera_stream
'''
Jetson Nano (Görüntü İşleme + Kontrol) 
        ↓ UDP
Arduino / STM32 (UDP dinler + SBUS üretir)
        ↓ SBUS
KX Flight Controller (Kumanda sinyali gibi kontrol alır)

'''
# ======== UDP Ayarları ========
UDP_IP = "192.168.144.50"   # Arduino/STM32 tarafının IP'si
UDP_PORT = 14550

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ======== Kontrol Fonksiyonu ========
def send_udp_command(throttle, roll, pitch, yaw):
    """
    4 kanal kontrol komutunu UDP ile gönder.
    Değerler genellikle 1000–2000 arasında olur.
    """
    message = f"{throttle},{roll},{pitch},{yaw}"
    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

# ======== Otonomi Akışı ========
def autonomous_flight():
    print("[INFO] ARM edildi, kalkış başlıyor...")

    # 1. KALKIŞ (10 saniye boyunca throttle artır)
    for _ in range(30):  # 30 frame boyunca yüksel
        send_udp_command(throttle=1700, roll=1500, pitch=1500, yaw=1500)
        time.sleep(0.3)

    print("[INFO] Tarama başlıyor...")
    scan_mode = True
    detected_hexagon = None
    scan_count = 0

    # 2. TARAMA
    while scan_mode:
        frame = get_camera_frame()
        if frame is None:
            continue

        result = process_frame(frame, area_threshold=800, debug=True)
        detections = result.get("detections", [])

        for det in detections:
            if det["shape"] == "hexagon" and det["color"] == "blue":
                print(f"[INFO] Altıgen bulundu: {det['center']}")
                detected_hexagon = det
                scan_mode = False
                break

        # Basit sağ-sol tarama (yaw hareketi)
        if scan_count % 20 < 10:
            send_udp_command(throttle=1500, roll=1500, pitch=1500, yaw=1600)
        else:
            send_udp_command(throttle=1500, roll=1500, pitch=1500, yaw=1400)

        scan_count += 1
        time.sleep(0.3)

    # 3. ALTIGEN MERKEZİNE YÖNELME
    print("[INFO] Hedefe yönelme başlıyor...")
    for _ in range(25):
        cx, cy = detected_hexagon["center"]

        # Basit yönlendirme mantığı (merkez ekran: 320x240)
        if cx < 280:
            yaw = 1400
        elif cx > 360:
            yaw = 1600
        else:
            yaw = 1500

        if cy < 200:
            pitch = 1600
        elif cy > 280:
            pitch = 1400
        else:
            pitch = 1500

        send_udp_command(throttle=1500, roll=1500, pitch=pitch, yaw=yaw)
        time.sleep(0.2)

    # 4. ALÇALMA
    print("[INFO] Hedefteyiz. Alçalma başlatıldı...")
    for _ in range(30):
        send_udp_command(throttle=1300, roll=1500, pitch=1500, yaw=1500)
        time.sleep(0.3)

    # 5. Motorları kapat (throttle=1000)
    print("[INFO] Görev tamamlandı. Motor kapatılıyor...")
    send_udp_command(throttle=1000, roll=1500, pitch=1500, yaw=1500)

# ========== BAŞLAT ==========
if __name__ == "__main__":
    try:
        autonomous_flight()
    except KeyboardInterrupt:
        print("\n[EXIT] Manuel durduruldu.")
        send_udp_command(throttle=1000, roll=1500, pitch=1500, yaw=1500)
