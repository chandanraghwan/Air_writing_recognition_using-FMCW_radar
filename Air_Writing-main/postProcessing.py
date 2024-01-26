import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import cv2

df = pd.read_csv(r"E:\airwriting\Dataset\2D-Range-Doppler Data\Digit_4\Digit_4_1.csv", header=None)

def complex_str_to_complex(complex_str):
    parts = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', complex_str)
    if len(parts) >= 2:
        real_part = float(parts[0])
        imag_part = float(parts[1])
    else:
        real_part = float(parts[0])
        imag_part = 0.0
    return complex(real_part, imag_part)

data = df.applymap(complex_str_to_complex).values
magnitude_spectrogram = np.abs(data)

min_output = 0
max_output = 255
min_input = np.min(magnitude_spectrogram)
max_input = np.max(magnitude_spectrogram)

scaled_value = (magnitude_spectrogram - min_input) / (max_input - min_input) * (max_output - min_output) + min_output

epsilon = 1e-3
log_scaled_value = 20 * np.log10(scaled_value + epsilon)

min_log_value = np.min(log_scaled_value)
max_log_value = np.max(log_scaled_value)

normalized_value = (log_scaled_value - min_log_value) / (max_log_value - min_log_value)

print(normalized_value.shape)
normalized_image = (normalized_value * 255).astype(np.uint8)

cv2.imwrite("4_1_maded.png", normalized_image)

plt.imshow(normalized_image, aspect='auto')
plt.show()
