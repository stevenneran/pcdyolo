import subprocess
import sys
import streamlit as st
from PIL import Image
import numpy as np

# Fungsi untuk menginstal modul dari requirements.txt
def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Instalasi dari requirements.txt
install_requirements()

# Mengimpor modul setelah instalasi
import cv2
from ultralytics import YOLO

# Fungsi untuk menjalankan model YOLO dan menampilkan hasilnya
def run_yolo(model, image_path):
    model = YOLO(model)
    results = model.predict(image_path, save=False, imgsz=320, conf=0.5)
    return results

# Judul aplikasi
st.title('Deteksi Jamur pada Roti dengan YOLOv8')

# Upload gambar
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menyimpan file yang diupload ke disk
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Menjalankan model YOLO pada gambar yang diupload
    results_1 = run_yolo("best.pt", "uploaded_image.jpg")
    results_2 = run_yolo("best2.pt", "uploaded_image.jpg")

    # Mengambil gambar yang dihasilkan dari hasil deteksi
    detected_img_1 = results_1[0].plot()
    detected_img_2 = results_2[0].plot()

    # Mengubah gambar hasil deteksi ke format yang bisa ditampilkan oleh Streamlit
    detected_img_pil_1 = Image.fromarray(detected_img_1)
    detected_img_pil_2 = Image.fromarray(detected_img_2)

    # Menampilkan hasil deteksi
    st.image(detected_img_pil_1, caption='Hasil Deteksi dengan Model 1', use_column_width=True)
    st.image(detected_img_pil_2, caption='Hasil Deteksi dengan Model 2', use_column_width=True)
