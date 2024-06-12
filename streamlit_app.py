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
def run_yolo(image_path):
    model = YOLO('best.pt')  # Ganti dengan model yang Anda gunakan
    results = model.predict(image_path, save=False, imgsz=320, conf=0.5)
    return results

# Judul aplikasi
st.title('YOLO Object Detection Dashboard')

# Upload gambar
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menyimpan file yang diupload ke disk
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Menjalankan model YOLO pada gambar yang diupload
    results = run_yolo("uploaded_image.jpg")

    # Mengambil gambar yang dihasilkan dari hasil deteksi
    detected_img = results[0].plot()  # Assuming results is a list of results

    # Mengubah gambar hasil deteksi ke format yang bisa ditampilkan oleh Streamlit
    detected_img_pil = Image.fromarray(detected_img)

    # Menampilkan hasil deteksi
    st.image(detected_img_pil, caption='Detected Image', use_column_width=True)
