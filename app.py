import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from fpdf import FPDF

# Fungsi untuk memuat model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model_data = pickle.load(file)
    return model_data

# Fungsi untuk memproses gambar dan mengekstrak fitur
def preprocess_image_all_features(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Gambar tidak valid atau tidak ditemukan.")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_rgb = np.mean(image_rgb, axis=(0, 1))
        mean_hsv = np.mean(image_hsv, axis=(0, 1))
        features = {
            'Red': mean_rgb[0],
            'Green': mean_rgb[1],
            'Blue': mean_rgb[2],
            'Hue': mean_hsv[0],
            'Saturation': mean_hsv[1],
            'Value': mean_hsv[2],
        }
        return features
    except Exception as e:
        st.error(f"Error in preprocess_image_all_features: {e}")
        return None

# Fungsi untuk memprediksi dosis
def predict_dose(image_path, model, significant_features):
    try:
        features = preprocess_image_all_features(image_path)
        if features is None:
            raise ValueError("Fitur gambar tidak valid.")
        selected_features = [features[feature] for feature in significant_features]
        dose = model.predict([selected_features])[0]
        return dose, features
    except Exception as e:
        st.error(f"Error in predict_dose: {e}")
        return None, None

# Fungsi untuk membuat PDF laporan
def generate_pdf(sample_name, dose, features):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Laporan Pembacaan Dosimeter", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Nama Sampel: {sample_name}", ln=True)
    pdf.cell(200, 10, txt=f"Dosis yang Diprediksi: {dose:.2f} Gy", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Fitur Warna:", ln=True)
    for key, value in features.items():
        pdf.cell(200, 10, txt=f"{key}: {value:.2f}", ln=True)
    file_path = "laporan_dosimeter.pdf"
    pdf.output(file_path)
    return file_path

# Fungsi untuk visualisasi analisis fitur
def visualize_features(dataset):
    try:
        features = ['Green', 'Blue', 'Saturation']
        for feature in features:
            x = dataset[feature].values.reshape(-1, 1)
            y = dataset['Dose']

            # Membuat model regresi linear
            model = LinearRegression()
            model.fit(x, y)

            # Plot hubungan fitur dengan dosis
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, color='blue', label='Data')
            plt.plot(x, model.predict(x), color='red', label='Regresi Linear')
            plt.title(f"Hubungan {feature} vs Dose")
            plt.xlabel(feature)
            plt.ylabel("Dose (Gy)")
            plt.legend()
            plt.grid()
            st.pyplot(plt)

            # Menampilkan nilai R-squared dan persamaan regresi
            r2_score = model.score(x, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            st.write(f"**Feature: {feature}**")
            st.write(f"R-squared: **{r2_score:.2f}**")
            st.write(f"Persamaan Regresi: Dose = {slope:.2f} * {feature} + {intercept:.2f}")
    except Exception as e:
        st.error(f"Error in visualize_features: {e}")

# Main function untuk aplikasi
def main():
    st.sidebar.title("Menu Navigasi")
    menu = st.sidebar.selectbox("Pilih Menu", ["Beranda", "Unggah Sampel", "Analisis Fitur", "History", "Tentang"])


    # Menu Beranda
    if menu == "Beranda":
        st.title("Aplikasi Dosimeter Film Reader")
        st.write("""
            Selamat datang di aplikasi Dosimeter Film Reader! Aplikasi ini dirancang untuk membaca film dosimeter menggunakan scanner standar dan teknologi machine learning.
        """)
        st.subheader("Siapa yang dapat menggunakan aplikasi ini?")
        st.write("""
            - **Laboratorium Radiasi**: Untuk memonitor paparan radiasi di lingkungan kerja.
            - **Rumah Sakit dan Klinik Radiologi**: Untuk memantau dosis radiasi pasien dan staf medis.
            - **Universitas dan Institusi Pendidikan**: Untuk penelitian dan pembelajaran terkait dosimetri.
        """)
        st.subheader("Perusahaan penggunakan teknologi e-beam atau X-ray")
        st.write("""
            - **Industri Sterilisasi**: Untuk proses sterilisasi alat medis atau bahan makanan.
            - **Ekspor Makanan dan Buah**: Menggunakan teknologi e-beam atau X-ray untuk memastikan standar keamanan makanan sebelum diekspor.
            - **Industri Farmasi**: Menggunakan e-beam atau X-ray untuk sterilisasi produk farmasi.
        """)
        st.write("""
            Dengan pendekatan inovatif ini, aplikasi ini bertujuan untuk menyediakan solusi yang hemat biaya, efisien, dan mudah diakses bagi berbagai institusi.
        """)


    # Menu Unggah Sampel
    elif menu == "Unggah Sampel":
        st.title("Unggah Sampel")
        method = st.radio("Pilih metode analisis:", ["Pink Scheme", "Coming Soon"])
        if method == "Pink Scheme":
            model_data = load_model("model_random_forest.pkl")
            model = model_data['model']
            significant_features = model_data['features']
            uploaded_file = st.file_uploader("Unggah gambar dosimeter", type=['jpg', 'png'])
            if uploaded_file:
                with open("temp_image.jpg", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                dose, features = predict_dose("temp_image.jpg", model, significant_features)
                if dose is not None:
                    st.write(f"Dosis yang diprediksi: {dose:.2f} Gy")
                    st.write("Fitur warna yang diekstraksi:")
                    st.json(features)
                    sample_name = st.text_input("Masukkan nama sampel:", value="Sampel 1")
                    if st.button("Unduh Laporan PDF"):
                        pdf_path = generate_pdf(sample_name, dose, features)
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="Klik untuk mengunduh laporan",
                                data=pdf_file,
                                file_name="laporan_dosimeter.pdf",
                                mime="application/pdf",
                            )
        else:
            st.write("Metode ini akan segera hadir.")

    # Menu Analisis Fitur
    elif menu == "Analisis Fitur":
        st.title("Analisis Fitur")
        st.write("Visualisasi hubungan fitur warna dengan dosis radiasi.")
        dataset_path = "dataset_dosimeter.csv"
        try:
            dataset = pd.read_csv(dataset_path)
            st.write("Dataset yang dimuat:")
            st.dataframe(dataset.head())
            visualize_features(dataset)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    # Menu History
    elif menu == "History":
        st.title("Riwayat Pembacaan")
        history_file = "history.csv"
        if st.button("Muat Riwayat"):
            try:
                history_data = pd.read_csv(history_file)
                st.dataframe(history_data)
            except Exception as e:
                st.error("Belum ada data riwayat yang tersedia.")

    # Menu Tentang
    elif menu == "Tentang":
        st.title("Tentang Aplikasi")
        st.write("""
            Aplikasi ini dibuat oleh MAV Studio untuk membaca dosimeter film dan memprediksi dosis radiasi berdasarkan fitur warna menggunakan teknologi machine learning. Versi 1.0.
        """)

if __name__ == "__main__":
    main()
