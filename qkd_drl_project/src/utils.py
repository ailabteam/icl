# src/utils.py

import numpy as np
from skyfield.api import load, EarthSatellite, Topos

# ==============================================================================
# 1. CONSTANTS AND SYSTEM PARAMETERS
# ==============================================================================

# -- QKD System Parameters
QKD_PARAMS = {
    'pulse_rate': 1e9,
    'mean_photon_number': 0.5,
    'decoy_mean_photon_number': 0.1,
    'detector_efficiency': 0.7,
    'dark_count_prob': 1e-7,
    'error_correction_eff': 1.15,
    'optical_alignment_error': 0.03,
}

# -- Simulation Parameters
SIM_PARAMS = {
    'min_elevation': 10.0,
}


# ==============================================================================
# 2. SECRET KEY RATE CALCULATION
# ==============================================================================

def calculate_skr(distance_km, zenith_angle_rad):
    """
    Tính toán Secret Key Rate (SKR) dựa trên mô hình suy hao kênh đơn giản hóa.
    SKR = R_0 * Transmittance, trong đó R_0 là SKR cơ sở ở điều kiện lý tưởng.
    """
    # R_0: SKR cơ sở (bps) ở khoảng cách 0. 
    # Giá trị này đại diện cho hiệu năng tối đa của hệ thống phần cứng QKD.
    # 10 Mbps là một con số lạc quan nhưng hợp lý cho hệ thống thế hệ mới.
    R_0 = 10e6 # 10 Mbps

    # a. Transmittance Calculation
    distance_m = distance_km * 1000

    # Suy hao khí quyển: T_atm = exp(-alpha / cos(zenith))
    # Đây là mô hình Beer-Lambert chuẩn.
    alpha_atm = 0.4 # Hệ số suy hao zenith (khá trong)
    if np.cos(zenith_angle_rad) <= 1e-6:
        return 0.0
    T_atm = np.exp(-alpha_atm / np.cos(zenith_angle_rad))
    
    # Suy hao không gian tự do (FSPL): T_fspl = (lambda / (4 * pi * d))^2
    # Để đơn giản, ta mô hình hóa nó như một hệ số suy giảm.
    # (A_rx / (pi * (d * tan(theta_div))^2))
    # Gộp tất cả các yếu tố quang học (bước sóng, khẩu độ,...) vào một hằng số.
    # Dựa trên các tài liệu, suy hao cho liên kết LEO-Mặt đất thường là -40 đến -60 dB.
    # Tương đương với độ truyền qua 10^-4 đến 10^-6.
    # Ta dùng một mô hình 1/d^2 đơn giản với hệ số phù hợp.
    fspl_coeff = 1e10 # Hệ số này cần được điều chỉnh để cho ra kết quả hợp lý
    T_fspl = fspl_coeff / (distance_m**2)
    
    # Hiệu suất hệ thống (máy dò, quang học,...)
    eta_sys = QKD_PARAMS['detector_efficiency'] # 0.7
    
    transmittance = T_atm * T_fspl * eta_sys
    
    skr = R_0 * transmittance
    
    return max(0.0, skr)


# ==============================================================================
# 3. HELPER FUNCTIONS FOR SIMULATION SETUP
# ==============================================================================

def create_constellation(ts, num_satellites, altitude_km=550):
    """
    Tạo một chòm sao vệ tinh LEO đơn giản.
    """
    satellites = []
    line1_template = '1 {:05d}U 23001A   23001.00000000  .00000000  00000-0  50000-4 0  9999'
    line2_template = '2 {:05d}  97.6000 {:08.4f} 0001000  30.0000 {:08.4f} 15.20000000'

    for i in range(num_satellites):
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
    """
    stations = {}
    for name, (lat, lon, elev) in locations.items():
        stations[name] = Topos(latitude_degrees=lat, longitude_degrees=lon, elevation_m=elev)
    return stations


# ==============================================================================
# 4. MAIN BLOCK FOR DEBUGGING
# ==============================================================================

if __name__ == '__main__':
    print("--- DEBUGGING REVISED calculate_skr FUNCTION ---")
    
    # Kịch bản 1: Khoảng cách gần, góc nâng cao
    print("\n[Test Case 1]: Ideal conditions")
    distance_ideal = 600
    zenith_ideal = np.deg2rad(5)
    skr_ideal = calculate_skr(distance_ideal, zenith_ideal)
    print(f"Distance: {distance_ideal} km, Zenith: {np.rad2deg(zenith_ideal):.2f} deg -> SKR: {skr_ideal / 1e3:.2f} kbps")
    if skr_ideal > 1000: # Kỳ vọng SKR cao
        print("  -> PASSED: SKR is positive and high as expected.")
    else:
        print(f"  -> FAILED: SKR is too low ({skr_ideal / 1e3:.2f} kbps). Check coefficients.")

    # Kịch bản 2: Khoảng cách xa, góc nâng thấp
    print("\n[Test Case 2]: Marginal conditions")
    distance_marginal = 2500
    zenith_marginal = np.deg2rad(80)
    skr_marginal = calculate_skr(distance_marginal, zenith_marginal)
    print(f"Distance: {distance_marginal} km, Zenith: {np.rad2deg(zenith_marginal):.2f} deg -> SKR: {skr_marginal:.2f} bps")
    if skr_marginal < 1000 and skr_marginal >= 0: # Kỳ vọng SKR thấp nhưng có thể > 0
        print("  -> PASSED: SKR is low or zero as expected.")
    else:
        print("  -> FAILED: SKR has an unexpected value.")

    # Kịch bản 3: Điều kiện thực tế
    print("\n[Test Case 3]: Realistic Pass conditions")
    distance_realistic = 1200
    zenith_realistic = np.deg2rad(60)
    skr_realistic = calculate_skr(distance_realistic, zenith_realistic)
    print(f"Distance: {distance_realistic} km, Zenith: {np.rad2deg(zenith_realistic):.2f} deg -> SKR: {skr_realistic / 1e3:.2f} kbps")
    if skr_realistic > 0:
        print("  -> PASSED: SKR is positive for realistic pass.")
    else:
        print("  -> FAILED: SKR is zero, which might indicate an issue.")
