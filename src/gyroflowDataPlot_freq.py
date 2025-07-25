import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import tkinter as tk
from tkinter import filedialog

# Load CSV file
# 打开文件选择对话框
root = tk.Tk()
root.withdraw()  # 不显示主窗口
file_path = filedialog.askopenfilename(
    title="请选择一个CSV文件",
    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
)
# 如果用户取消选择，file_path 会是空字符串
if not file_path:
    print("未选择文件，程序退出。")
    exit()
# df = pd.read_csv("data/DJI_20250723171817_0002_D.csv")
# 然后用 pandas 读取这个文件
df = pd.read_csv(file_path)

# Convert timestamp from ms to seconds
df["Time"] = df["timestamp_ms"] / 1000.0

# Select data within 6s to 7s
df_slice = df[(df["Time"] >= 6.0) & (df["Time"] < 7.0)]

# Extract time and angular positions
time = df_slice["Time"].values
pitch = df_slice["org_pitch"].values
yaw = df_slice["org_yaw"].values
roll = df_slice["org_roll"].values

# Compute time step
dt = np.mean(np.diff(time))

# Compute angular velocity (dθ/dt)
angular_velocity_pitch = np.gradient(pitch, time)
angular_velocity_yaw = np.gradient(yaw, time)
angular_velocity_roll = np.gradient(roll, time)

# Compute angular acceleration (d²θ/dt²)
angular_accel_pitch = np.gradient(angular_velocity_pitch, time)
angular_accel_yaw = np.gradient(angular_velocity_yaw, time)
angular_accel_roll = np.gradient(angular_velocity_roll, time)

# Perform FFT
N = len(time)  # Number of samples
freqs = np.fft.fftfreq(N, d=dt)  # Frequency bin

window = np.hanning(N)

# Apply window before FFT
fft_pitch = np.abs(fft(angular_accel_pitch * window))[:N//2] / N
fft_yaw   = np.abs(fft(angular_accel_yaw * window))[:N//2] / N
fft_roll  = np.abs(fft(angular_accel_roll * window))[:N//2] / N


# fft_pitch = np.abs(fft(angular_accel_pitch))[:N//2]  # Magnitude spectrum
# fft_yaw = np.abs(fft(angular_accel_yaw))[:N//2]
# fft_roll = np.abs(fft(angular_accel_roll))[:N//2]
freqs = freqs[:N//2]  # Keep only positive frequencies

# Plot FFT of Angular Acceleration
plt.figure(figsize=(10, 6))
plt.plot(freqs, fft_pitch, label="Pitch", color="r")
plt.plot(freqs, fft_yaw, label="Yaw", color="g")
plt.plot(freqs, fft_roll, label="Roll", color="b")
plt.xlim(0, 200)  # Limit to 0-100 Hz
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of Angular Acceleration (6-7s)")
plt.legend()
plt.grid()
plt.show()
