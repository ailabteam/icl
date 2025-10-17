# src/utils.py

import numpy as np
from skyfield.api import load, EarthSatellite, Topos

# ==============================================================================
# 1. CONSTANTS AND SYSTEM PARAMETERS
# ==============================================================================

# -- QKD System Parameters (Dựa trên các giá trị thực tế trong tài liệu tham khảo)
# Tham khảo: "Satellite-based entanglement distribution with aerial platforms"
# và các bài báo tương tự.
QKD_PARAMS = {
    'pulse_rate': 1e9,  # Tốc độ lặp laser (Hz)
    'mean_photon_number': 0.5,  # Số photon trung bình mỗi xung (tín hiệu)
    'decoy_mean_photon_number': 0.1,  # Số photon trung bình mỗi xung (mồi)
    'detector_efficiency': 0.7,  # Hiệu suất của máy dò photon đơn
    'dark_count_prob': 1e-7,  # Xác suất đếm tối trên mỗi cổng
    'error_correction_eff': 1.15,  # Hiệu quả của mã sửa lỗi
    'optical_alignment_error': 0.03, # Tỷ lệ lỗi do căn chỉnh quang học (e_d)
}

# -- Simulation Parameters
SIM_PARAMS = {
    'min_elevation': 10.0, # Góc nâng tối thiểu để có kết nối (độ)
}


# ==============================================================================
# 2. SECRET KEY RATE CALCULATION
# ==============================================================================

def calculate_skr(distance_km, zenith_angle_rad):
    """
    Tính toán Secret Key Rate (SKR) cho giao thức decoy-state BB84.
    Mô hình này là phiên bản đơn giản hóa dựa trên các công thức chuẩn.

    Args:
        distance_km (float): Khoảng cách từ vệ tinh đến trạm mặt đất (km).
        zenith_angle_rad (float): Góc thiên đỉnh tại trạm mặt đất (radians).

    Returns:
        float: Secret Key Rate (bits per second).
    """
    # Chuyển đổi khoảng cách sang mét
    distance_m = distance_km * 1000

    # a. Tính toán độ truyền qua (Transmittance)
    # Transmittance = T_atm * T_fspl
    H = 8.0  # Độ cao hiệu dụng của khí quyển (km)
    h = 0    # Độ cao của trạm mặt đất so với mực nước biển (km)
    alpha = 0.4 # Hệ số suy hao khí quyển (dB/km ở zenith)
    
    # Suy hao khí quyển
    L_atm_db = alpha * (np.exp((h - H) / H) - np.exp(-distance_km / H)) / np.cos(zenith_angle_rad) if np.cos(zenith_angle_rad) > 0 else float('inf')
    T_atm = 10**(-L_atm_db / 10)
    
    # Suy hao không gian tự do (Free-Space Path Loss) - Giả định đơn giản
    # Trong thực tế, nó phụ thuộc vào khẩu độ kính thiên văn, bước sóng.
    # Ở đây, chúng ta dùng một mô hình suy giảm theo khoảng cách đơn giản hơn.
    # T_fspl = (lambda / (4 * pi * d))^2
    # Giả định một hệ số T_link_base để đại diện cho các yếu tố quang học cố định
    T_link_base = 1e-5 # Giá trị này cần được hiệu chỉnh dựa trên thiết kế hệ thống
    T_fspl = T_link_base / (distance_m**2)
    
    transmittance = T_atm * T_fspl * QKD_PARAMS['detector_efficiency']
    
    if transmittance <= 0:
        return 0.0

    # b. Tính toán Quantum Bit Error Rate (QBER)
    e_0 = 0.5 # Tỷ lệ lỗi của photon nền
    Y_0 = 2 * QKD_PARAMS['dark_count_prob'] # Tỷ lệ đếm nền
    mu = QKD_PARAMS['mean_photon_number']
    
    qber_numerator = e_0 * Y_0 + QKD_PARAMS['optical_alignment_error'] * mu * transmittance
    qber_denominator = Y_0 + mu * transmittance
    
    qber = qber_numerator / qber_denominator if qber_denominator > 0 else QKD_PARAMS['optical_alignment_error']

    # c. Tính toán Tỷ lệ sàng lọc (Sifted Key Rate)
    # Đối với BB84, tỷ lệ sàng lọc là 0.5
    sift_rate = 0.5 * QKD_PARAMS['pulse_rate'] * (Y_0 + mu * transmittance)

    # d. Tính toán SKR (sử dụng công thức GLLP)
    h2 = lambda x: -x * np.log2(x) - (1 - x) * np.log2(1 - x) if 0 < x < 1 else 0
    
    privacy_amplification_term = (1 - h2(qber))
    error_correction_term = QKD_PARAMS['error_correction_eff'] * h2(qber)
    
    skr = sift_rate * (privacy_amplification_term - error_correction_term)
    
    return max(0.0, skr)


# ==============================================================================
# 3. HELPER FUNCTIONS FOR SIMULATION SETUP
# ==============================================================================

def create_constellation(ts, num_satellites, altitude_km=550):
    """
    Tạo một chòm sao vệ tinh LEO đơn giản.
    Để đơn giản, chúng ta phân bố chúng đều trên một mặt phẳng quỹ đạo gần cực.
    """
    satellites = []
    line1_template = '1 {:05d}U 23001A   23001.00000000  .00000000  00000-0  50000-4 0  9999'
    line2_template = '2 {:05d}  97.6000 {:08.4f} 0001000  30.0000 {:08.4f} 15.20000000'
    
    for i in range(num_satellites):
        # Phân bố đều các vệ tinh trên quỹ đạo
        right_ascension = (360.0 / num_satellites) * i
        mean_anomaly = (360.0 / num_satellites) * i 
        
        line1 = line1_template.format(i)
        line2 = line2_template.format(i, right_ascension, mean_anomaly)
        
        satellite = EarthSatellite(line1, line2, f'SAT-{i}', ts)
        satellites.append(satellite)
        
    return satellites

def create_ground_stations(ts, locations):
    """
    Tạo các đối tượng trạm mặt đất từ danh sách tọa độ.
    locations: dict of {'name': (lat, lon, elev_m)}
    """
    stations = {}
    for name, (lat, lon, elev) in locations.items():
        stations[name] = Topos(latitude_degrees=lat, longitude_degrees=lon, elevation_m=elev)
    return stations


# ==============================================================================
# 4. MAIN BLOCK FOR TESTING
# ==============================================================================

if __name__ == '__main__':
    # Đoạn code này chỉ chạy khi bạn thực thi file trực tiếp: python src/utils.py
    # Dùng để kiểm tra nhanh các hàm bên trên có hoạt động đúng không.
    
    ts = load.timescale()
    t = ts.now()

    # --- Tạo vệ tinh và trạm mặt đất
    sats = create_constellation(ts, num_satellites=1)
    sat = sats[0]
    
    # Vị trí của Hà Nội
    stations = create_ground_stations(ts, {'Hanoi': (21.0285, 105.8542, 20)})
    hanoi = stations['Hanoi']
    
    # --- Tính toán hình học
    difference = sat - hanoi
    topocentric = difference.at(t)
    
    alt, az, distance = topocentric.altaz()
    zenith_rad = np.deg2rad(90.0 - alt.degrees)

    print(f"Kiểm tra hàm tiện ích tại thời điểm: {t.utc_strftime()}")
    print("-" * 30)
    print(f"Vệ tinh: {sat.name}")
    print(f"Trạm mặt đất: Hanoi")
    print(f"Góc nâng (Elevation): {alt.degrees:.2f} độ")
    print(f"Khoảng cách: {distance.km:.2f} km")
    print(f"Góc thiên đỉnh (Zenith): {np.rad2deg(zenith_rad):.2f} độ")

    # --- Kiểm tra tính toán SKR
    if alt.degrees > SIM_PARAMS['min_elevation']:
        skr = calculate_skr(distance.km, zenith_rad)
        print(f"--> LIÊN KẾT KHẢ DỤNG!")
        print(f"--> Secret Key Rate (SKR) ước tính: {skr / 1e3:.2f} kbps")
    else:
        print(f"--> Liên kết không khả dụng (góc nâng < {SIM_PARAMS['min_elevation']} độ).")
        skr = calculate_skr(distance.km, zenith_rad)
        print(f"--> SKR (dự kiến là 0): {skr:.2f} bps")
