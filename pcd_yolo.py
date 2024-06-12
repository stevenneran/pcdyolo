import subprocess
import sys
import streamlit as st
from PIL import Image
import numpy as np
import base64

# Fungsi untuk menginstal modul dari requirements.txt
def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Instalasi dari requirements.txt
install_requirements()

# Mengimpor library
import cv2
from ultralytics import YOLO

# Fungsi untuk menjalankan model YOLO dan menampilkan hasilnya
def run_yolo(model, image_path):
    model = YOLO(model)
    results = model.predict(image_path, save=False, imgsz=320, conf=0.5)
    return results

# Fungsi untuk mengonversi citra BGR ke RGB
def convert_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Fungsi untuk menyisipkan CSS ke dalam halaman Streamlit
def add_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Menambahkan background dari file background.png
add_background("background2.png")

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

    # Mengonversi gambar dari BGR ke RGB
    detected_img_1_rgb = convert_bgr_to_rgb(detected_img_1)
    detected_img_2_rgb = convert_bgr_to_rgb(detected_img_2)

    # Mengubah gambar hasil deteksi ke format yang bisa ditampilkan oleh Streamlit
    detected_img_pil_1 = Image.fromarray(detected_img_1_rgb)
    detected_img_pil_2 = Image.fromarray(detected_img_2_rgb)

    # Membaca gambar asli
    original_img = Image.open("uploaded_image.jpg")

    # Membuat dua kolom untuk menampilkan gambar asli dan gambar hasil deteksi secara berdampingan
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original_img, caption='Citra Asli', use_column_width=True)

    with col2:
        st.image(detected_img_pil_1, caption='Hasil Deteksi dengan Model 1', use_column_width=True)

    with col3:
        st.image(detected_img_pil_2, caption='Hasil Deteksi dengan Model 2', use_column_width=True)
