import os
import cv2
import numpy as np
import pandas as pd

def extract_rgb_hsv_from_images(data_dir):
    data = []
    labels = []
    
    # Iterasi setiap folder kategori dosis
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            dose_label = float(category.split('_')[0])  # Ambil dosis dari nama folder
            
            # Iterasi setiap file gambar
            for file_name in os.listdir(category_path):
                file_path = os.path.join(category_path, file_name)
                if file_name.endswith(('.jpg', '.png')):
                    # Baca gambar
                    image = cv2.imread(file_path)
                    # Konversi ke RGB dan HSV
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    # Hitung nilai rata-rata RGB dan HSV
                    mean_rgb = np.mean(image_rgb, axis=(0, 1))
                    mean_hsv = np.mean(image_hsv, axis=(0, 1))
                    # Gabungkan RGB dan HSV
                    features = np.concatenate((mean_rgb, mean_hsv))
                    data.append(features)
                    labels.append(dose_label)
    
    # Buat DataFrame
    columns = ['Red', 'Green', 'Blue', 'Hue', 'Saturation', 'Value']
    df = pd.DataFrame(data, columns=columns)
    df['Dose'] = labels
    return df

# Jalankan preprocessing
if __name__ == "__main__":
    data_dir = "Data/Hasil_Scan"
    dataset = extract_rgb_hsv_from_images(data_dir)
    dataset.to_csv('dataset_dosimeter.csv', index=False)
    print("Dataset berhasil disimpan ke 'dataset_dosimeter.csv'.")
