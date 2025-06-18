from pymavlink import mavutil
import time
from modules.vision_engine import process_frame
from modules.camera_input import get_camera_stream
'''
Jetson Nano (Görüntü işleme + karar) 
      ↓ MAVLink (via UDP veya Serial)
Pixhawk (uçuş kontrolcü) 
      → Motor, yön, kalkış, iniş vs.
'''
# MAVLink bağlantısı kur (UDP veya serial)
master = mavutil.mavlink_connection('udpout:127.0.0.1:14550')
master.wait_heartbeat()
print(f"[INFO] Bağlandı: {master.target_system}")

def arm_and_takeoff(target_altitude):
    # ARM
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)

    master.set_mode_apm("GUIDED")
    print("[INFO] ARM edildi, GUIDED moda geçildi.")
    time.sleep(2)

    # Kalkış
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, target_altitude)
    print(f"[INFO] Kalkış başlatıldı: {target_altitude} m")

    time.sleep(10)  # Yükselmeyi bekle

def goto_target_ned(x, y, z):
    master.mav.set_position_target_local_ned_send(
        int(round(time.time() * 1e6)),
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,
        x, y, z,
        0, 0, 0,
        0, 0, 0,
        0, 0)

def land():
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0)
    print("[INFO] İniş başlatıldı.")

# ======== Ana Otonom Uçuş Mantığı ========
def autonomous_mission():
    arm_and_takeoff(target_altitude=10)

    print("[INFO] Tarama başlıyor...")
    detected = None
    start_time = time.time()

    while time.time() - start_time < 20:  # 20 saniyelik tarama
        frame = get_camera_frame()
        if frame is None:
            continue

        result = process_frame(frame, area_threshold=800, debug=True)
        for det in result.get("detections", []):
            if det["shape"] == "hexagon" and det["color"] == "blue":
                detected = det
                print(f"[INFO] Altıgen tespit edildi: {det['center']}")
                break

        if detected:
            break

    if detected:
        # Basit sabit NED yönelim — normalde GPS konum gerekir!
        print("[INFO] Hedefe yönelme...")
        goto_target_ned(x=5, y=0, z=-10)
        time.sleep(10)

        print("[INFO] Alçalma başlatıldı...")
        goto_target_ned(x=5, y=0, z=-1)
        time.sleep(5)

        land()
        print("[INFO] Görev tamamlandı.")

    else:
        print("[WARN] Hedef bulunamadı.")

# ========== BAŞLAT ==========

if __name__ == "__main__":
    try:
        autonomous_mission()
    except KeyboardInterrupt:
        print("\n[EXIT] Manuel durduruldu.")
