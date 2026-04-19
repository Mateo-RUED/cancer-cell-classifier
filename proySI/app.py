import streamlit as st
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


model = YOLO("best.pt")

st.title("🧬 Clasificador de células sanguíneas")
st.write("Subí una imagen microscópica para clasificar las células (blastocito, monocito, etc.)")


uploaded_file = st.file_uploader("Arrastrá o seleccioná una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    

    st.image(image, caption="🧫 Imagen cargada", use_column_width=True)
    

    results = model.predict(image)
    probs = results[0].probs  

    cls = probs.top1
    conf = float(probs.top1conf)
    st.write(f"### 🧬 Clase detectada: **{model.names[cls]}** con {conf*100:.2f}% de confianza")
    

    st.write("### 📊 Distribución de probabilidades por clase:")
    fig, ax = plt.subplots(figsize=(6, 3))
    nombres = list(model.names.values())
    valores = probs.data.cpu().numpy() * 100
    colores = ['#5DADE2' if i != cls else '#F4D03F' for i in range(len(nombres))]
    ax.barh(nombres, valores, color=colores)
    ax.set_xlabel("Confianza (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Probabilidades por clase")
    st.pyplot(fig)


    st.write("### 🔍 Visualización de la detección:")
    try:
        res_img = results[0].plot() 
        st.image(res_img, caption="Detección resaltada", use_column_width=True)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            xyxy = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            zoom = image.crop((x1, y1, x2, y2)).resize((256, 256))
            st.image(zoom, caption="🔎 Región de interés (zoom)", use_column_width=False)
    except Exception as e:
        st.warning(f"No se pudo generar la visualización detallada: {e}")
