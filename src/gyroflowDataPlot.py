import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
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

# Select data within 6s to 500s
df_slice = df[(df["Time"] >= 6.0) & (df["Time"] < 50.0)]

# Extract time and angular positions
time = df_slice["Time"].values
pitch = df_slice["org_pitch"].values
yaw = df_slice["org_yaw"].values
roll = df_slice["org_roll"].values

# Compute time differences
dt = np.gradient(time)

# Compute angular velocity (dθ/dt)
angular_velocity_pitch = np.gradient(pitch, time)
angular_velocity_yaw = np.gradient(yaw, time)
angular_velocity_roll = np.gradient(roll, time)

# Compute angular acceleration (d²θ/dt²)
angular_accel_pitch = np.gradient(angular_velocity_pitch, time)
angular_accel_yaw = np.gradient(angular_velocity_yaw, time)
angular_accel_roll = np.gradient(angular_velocity_roll, time)

# Sampling rate (Hz)
sampling_rate = 1 / np.mean(dt)

# Compute STFT (Short-Time Fourier Transform) using spectrogram
f_pitch, t_pitch, Sxx_pitch = spectrogram(angular_accel_pitch, fs=sampling_rate, nperseg=256)
f_yaw, t_yaw, Sxx_yaw = spectrogram(angular_accel_yaw, fs=sampling_rate, nperseg=256)
f_roll, t_roll, Sxx_roll = spectrogram(angular_accel_roll, fs=sampling_rate, nperseg=256)

# Limit to 0-100 Hz range
f_limit = 200
pitch_mask = f_pitch <= f_limit
yaw_mask = f_yaw <= f_limit
roll_mask = f_roll <= f_limit

# 手动设置颜色深度范围，例如统一为 [-100, -30] dB
vmin, vmax = -10, 80

# Plot Spectrograms for Angular Acceleration (0-100 Hz)
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].pcolormesh(t_pitch + 6, f_pitch[pitch_mask], 10 * np.log10(Sxx_pitch[pitch_mask, :]), shading='gouraud',cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_ylabel("Frequency (Hz)")
axes[0].set_title("STFT of Angular Acceleration - Pitch (0-100 Hz)")

axes[1].pcolormesh(t_yaw + 6, f_yaw[yaw_mask], 10 * np.log10(Sxx_yaw[yaw_mask, :]), shading='gouraud', cmap='viridis' , vmin=vmin, vmax=vmax)
axes[1].set_ylabel("Frequency (Hz)")
axes[1].set_title("STFT of Angular Acceleration - Yaw (0-100 Hz)")

axes[2].pcolormesh(t_roll + 6, f_roll[roll_mask], 10 * np.log10(Sxx_roll[roll_mask, :]), shading='gouraud',cmap='viridis' , vmin=vmin, vmax=vmax)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Frequency (Hz)")
axes[2].set_title("STFT of Angular Acceleration - Roll (0-100 Hz)")

plt.tight_layout()
plt.show()
