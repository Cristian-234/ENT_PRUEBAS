# =========================
# IMPORTS
# =========================
import streamlit as st
import cv2
import tempfile
import os
from PIL import Image
import tensorflow as tf
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Forzar compatibilidad con modelos antiguos de Keras si es necesario
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# =========================
# CONFIGURACIÓN DE STUN (SOLUCIONA EL ERROR DE CÁMARA)
# =========================
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]
})

# =========================
# CONFIGURACIÓN DE PÁGINA
# =========================
st.set_page_config(
    page_title="Detección de Enfermedades en Plantas",
    page_icon="🌱",
    layout="wide"
)

# =========================
# CONSTANTES
# =========================
IMG_SIZE = 224

labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
    'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomate___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

traducciones = [
    'Manzana — Sarna', 'Manzana — Podredumbre negra', 'Manzana — Roya del cedro', 'Manzana — Sana',
    'Fondo sin hojas', 'Arándano — Sano', 'Cereza — Mildiu polvoriento', 'Cereza — Sana',
    'Maíz — Mancha foliar', 'Maíz — Roya común', 'Maíz — Tizón norte', 'Maíz — Sano',
    'Uva — Podredumbre negra', 'Uva — Esca', 'Uva — Mancha foliar', 'Uva — Sana',
    'Naranja — Huanglongbing', 'Melocotón — Mancha bacteriana', 'Melocotón — Sano',
    'Pimiento — Mancha bacteriana', 'Pimiento — Sano',
    'Papa — Tizón temprano', 'Papa — Tizón tardío', 'Papa — Sana',
    'Frambuesa — Sana', 'Soja — Sana', 'Calabaza — Mildiu',
    'Fresa — Quemadura foliar', 'Fresa — Sana',
    'Tomate — Mancha bacteriana', 'Tomate — Tizón temprano', 'Tomate — Tizón tardío',
    'Tomate — Moho foliar', 'Tomate — Septoria',
    'Tomate — Ácaros', 'Tomate — Mancha objetivo',
    'Tomate — Virus rizo amarillo', 'Tomate — Virus mosaico',
    'Tomate — Sano'
]

# =========================
# INFO AGRONÓMICA
# =========================
info_enfermedades = {

    # 🍎 MANZANA
    "Apple___Apple_scab": {
        "planta": "Manzano",
        "descripcion": "Enfermedad fúngica que provoca manchas oscuras en hojas y frutos.",
        "recomendacion": "Aplicar fungicidas y realizar poda sanitaria."
    },
    "Apple___Black_rot": {
        "planta": "Manzano",
        "descripcion": "Produce pudrición negra en frutos y lesiones en hojas.",
        "recomendacion": "Eliminar frutos infectados y aplicar fungicidas."
    },
    "Apple___Cedar_apple_rust": {
        "planta": "Manzano",
        "descripcion": "Causa manchas amarillas y deformaciones foliares.",
        "recomendacion": "Controlar hospedantes alternos y aplicar fungicidas."
    },
    "Apple___healthy": {
        "planta": "Manzano",
        "descripcion": "La planta no presenta signos visibles de enfermedad.",
        "recomendacion": "Mantener buenas prácticas agrícolas."
    },

    # 🫐 ARÁNDANO
    "Blueberry___healthy": {
        "planta": "Arándano",
        "descripcion": "Planta sana sin síntomas visibles.",
        "recomendacion": "Continuar manejo preventivo."
    },

    # 🍒 CEREZA
    "Cherry___Powdery_mildew": {
        "planta": "Cereza",
        "descripcion": "Enfermedad fúngica que genera polvo blanco en hojas.",
        "recomendacion": "Aplicar fungicidas y mejorar ventilación."
    },
    "Cherry___healthy": {
        "planta": "Cereza",
        "descripcion": "Planta sana.",
        "recomendacion": "Mantener monitoreo regular."
    },

    # 🌽 MAÍZ
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "planta": "Maíz",
        "descripcion": "Manchas grises que reducen la fotosíntesis.",
        "recomendacion": "Rotación de cultivos y uso de fungicidas."
    },
    "Corn___Common_rust": {
        "planta": "Maíz",
        "descripcion": "Pústulas marrones en hojas.",
        "recomendacion": "Uso de variedades resistentes."
    },
    "Corn___Northern_Leaf_Blight": {
        "planta": "Maíz",
        "descripcion": "Lesiones alargadas que afectan el rendimiento.",
        "recomendacion": "Aplicar fungicidas y eliminar residuos."
    },
    "Corn___healthy": {
        "planta": "Maíz",
        "descripcion": "Cultivo sano.",
        "recomendacion": "Continuar manejo agronómico."
    },

    # 🍇 UVA
    "Grape___Black_rot": {
        "planta": "Uva",
        "descripcion": "Causa manchas negras y pudrición del fruto.",
        "recomendacion": "Aplicar fungicidas y eliminar restos infectados."
    },
    "Grape___Esca_(Black_Measles)": {
        "planta": "Uva",
        "descripcion": "Provoca necrosis interna y debilitamiento.",
        "recomendacion": "Eliminar plantas severamente afectadas."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "planta": "Uva",
        "descripcion": "Manchas foliares que reducen la producción.",
        "recomendacion": "Aplicar fungicidas preventivos."
    },
    "Grape___healthy": {
        "planta": "Uva",
        "descripcion": "Planta sin síntomas.",
        "recomendacion": "Mantener prácticas preventivas."
    },

    # 🍊 NARANJA
    "Orange___Haunglongbing_(Citrus_greening)": {
        "planta": "Naranja",
        "descripcion": "Enfermedad bacteriana grave que amarillea hojas.",
        "recomendacion": "Control del insecto vector y erradicación."
    },

    # 🍑 MELOCOTÓN
    "Peach___Bacterial_spot": {
        "planta": "Melocotón",
        "descripcion": "Manchas bacterianas en hojas y frutos.",
        "recomendacion": "Uso de bactericidas y poda."
    },
    "Peach___healthy": {
        "planta": "Melocotón",
        "descripcion": "Planta sana.",
        "recomendacion": "Manejo agrícola adecuado."
    },

    # 🌶️ PIMIENTO
    "Pepper,_bell___Bacterial_spot": {
        "planta": "Pimiento",
        "descripcion": "Manchas bacterianas que reducen calidad.",
        "recomendacion": "Rotación de cultivos y bactericidas."
    },
    "Pepper,_bell___healthy": {
        "planta": "Pimiento",
        "descripcion": "Planta sana.",
        "recomendacion": "Mantener monitoreo."
    },

    # 🥔 PAPA
    "Potato___Early_blight": {
        "planta": "Papa",
        "descripcion": "Manchas concéntricas en hojas.",
        "recomendacion": "Uso de fungicidas preventivos."
    },
    "Potato___Late_blight": {
        "planta": "Papa",
        "descripcion": "Provoca marchitez rápida y pudrición.",
        "recomendacion": "Eliminar plantas infectadas."
    },
    "Potato___healthy": {
        "planta": "Papa",
        "descripcion": "Cultivo sano.",
        "recomendacion": "Mantener manejo adecuado."
    },

    # 🍓 FRESA
    "Strawberry___Leaf_scorch": {
        "planta": "Fresa",
        "descripcion": "Manchas oscuras que queman hojas.",
        "recomendacion": "Eliminar hojas afectadas."
    },
    "Strawberry___healthy": {
        "planta": "Fresa",
        "descripcion": "Planta sana.",
        "recomendacion": "Continuar manejo preventivo."
    },

    # 🍅 TOMATE
    "Tomato___Bacterial_spot": {
        "planta": "Tomate",
        "descripcion": "Manchas bacterianas en hojas y frutos.",
        "recomendacion": "Uso de bactericidas."
    },
    "Tomato___Early_blight": {
        "planta": "Tomate",
        "descripcion": "Manchas marrones concéntricas.",
        "recomendacion": "Aplicar fungicidas."
    },
    "Tomato___Late_blight": {
        "planta": "Tomate",
        "descripcion": "Enfermedad grave de rápida propagación.",
        "recomendacion": "Eliminar plantas afectadas."
    },
    "Tomato___Leaf_Mold": {
        "planta": "Tomate",
        "descripcion": "Moho en el envés de las hojas.",
        "recomendacion": "Mejorar ventilación."
    },
    "Tomato___Septoria_leaf_spot": {
        "planta": "Tomate",
        "descripcion": "Manchas pequeñas con centros claros.",
        "recomendacion": "Aplicar fungicidas."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "planta": "Tomate",
        "descripcion": "Ácaros que causan amarillamiento.",
        "recomendacion": "Uso de acaricidas."
    },
    "Tomato___Target_Spot": {
        "planta": "Tomate",
        "descripcion": "Manchas circulares en hojas.",
        "recomendacion": "Control químico y cultural."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "planta": "Tomate",
        "descripcion": "Virus que causa enrollamiento foliar.",
        "recomendacion": "Control del insecto vector."
    },
    "Tomato___Tomato_mosaic_virus": {
        "planta": "Tomate",
        "descripcion": "Virus que provoca mosaico en hojas.",
        "recomendacion": "Eliminar plantas infectadas."
    },
    "Tomato___healthy": {
        "planta": "Tomate",
        "descripcion": "Planta sana.",
        "recomendacion": "Mantener buenas prácticas agrícolas."
    }
}

# =========================
# CARGA DEL MODELO
# =========================
@st.cache_resource
def load_model():
    # Cargamos el modelo sin compilar para evitar errores de serialización de optimizadores
    return tf.keras.models.load_model("modelo/bulbasaur.h5", compile=False)

# =========================
# PREPROCESAMIENTO
# =========================
def process_image_pil(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def process_frame_cv(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

# =========================
# VIDEO (Corregido para evitar MediaFileStorageError)
# =========================
def process_video(video_file, model):
    # Creamos un archivo temporal físico real en el disco del servidor
    # Esto evita el error de "Missing file" en la memoria de Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        temp_path = tfile.name

    cap = cv2.VideoCapture(temp_path)
    stframe = st.empty() # Espacio reservado para el video

    sanas, enfermas = 0, 0
    frame_count = 0

    # Barra de progreso para dar feedback al usuario
    progress_bar = st.progress(0)
    
    # Obtener total de frames para la barra de progreso
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Procesar 1 de cada 10 frames para no saturar la memoria y aumentar velocidad
        if frame_count % 10 != 0:
            continue

        # Actualizar progreso
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

        img = process_frame_cv(frame)
        pred = model.predict(img, verbose=0)
        label = np.argmax(pred)
        confidence = np.max(pred) * 100

        if "healthy" in labels[label]:
            text = f"SANA ({confidence:.2f}%)"
            color = (0, 255, 0)
            sanas += 1
        else:
            text = f"ENFERMA ({confidence:.2f}%)"
            color = (0, 0, 255)
            enfermas += 1

        # Dibujar en el frame
        h, w, _ = frame.shape
        cv2.rectangle(frame, (10, 10), (w-10, h-10), color, 3)
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Convertir y mostrar
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()
    progress_bar.empty()
    
    # Limpiar el archivo temporal del disco
    if os.path.exists(temp_path):
        os.remove(temp_path)

    st.subheader("📊 Resumen del video")
    col1, col2 = st.columns(2)
    col1.metric("Detecciones Sanas", sanas)
    col2.metric("Detecciones Enfermas", enfermas)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_norm = img_resized / 255.0
        img_input = np.expand_dims(img_norm, axis=0)

        pred = self.model.predict(img_input, verbose=0)
        label = np.argmax(pred)
        confidence = np.max(pred) * 100

        if "healthy" in labels[label]:
            text = f"SANA ({confidence:.1f}%)"
            color = (0, 255, 0)
        else:
            text = f"ENFERMA ({confidence:.1f}%)"
            color = (0, 0, 255)

        cv2.putText(img, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(img, (10, 10), (img.shape[1]-10, img.shape[0]-10), color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================
# APP PRINCIPAL
# =========================
def main():
    st.markdown("""
    <h1 style='text-align:center; color:#2e7d32;'>🌱 Sistema Inteligente de Detección de Enfermedades en Plantas</h1>
    <h4 style='text-align:center;'>Deep Learning aplicado a la agricultura</h4>
    <hr>
    """, unsafe_allow_html=True)

    # =========================
    # SIDEBAR
    # =========================
    st.sidebar.title("📋 Panel de Control")
    opcion = st.sidebar.radio(
        "Modo de análisis",
        ("Imagen", "Video", "Cámara en tiempo real")
    )

    st.sidebar.markdown("### 🌿 Cultivos soportados")
    st.sidebar.write(
        "Manzana, Maíz, Uva, Tomate, Papa, Fresa, Cítricos, Pimiento"
    )

    model = load_model()

    # =========================
    # MODO IMAGEN
    # =========================
    if opcion == "Imagen":
        file = st.file_uploader(
            "📷 Carga una imagen",
            type=["jpg", "png", "jpeg"]
        )

        if file:
            image = Image.open(file)
            st.image(image, use_column_width=True)

            img = process_image_pil(image)
            pred = model.predict(img, verbose=0)
            label = np.argmax(pred)
            confidence = np.max(pred) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Estado",
                "Sana" if "healthy" in labels[label] else "Enferma"
            )
            col2.metric("Confianza", f"{confidence:.2f}%")
            col3.metric("Clase", traducciones[label])

            key = labels[label]
            if key in info_enfermedades:
                st.subheader("📖 Información agronómica")
                st.write(f"**Planta:** {info_enfermedades[key]['planta']}")
                st.write(f"**Descripción:** {info_enfermedades[key]['descripcion']}")
                st.write(f"**Recomendación:** {info_enfermedades[key]['recomendacion']}")

    # =========================
    # MODO VIDEO
    # =========================
    elif opcion == "Video":
        video = st.file_uploader(
            "🎥 Carga un video",
            type=["mp4", "avi", "mov"]
        )

        if video:
            st.info("Procesando video...")
            process_video(video, model)

    # =========================
    # MODO CÁMARA EN TIEMPO REAL
    # =========================
    elif opcion == "Cámara en tiempo real":
        st.subheader("🎦 Detección en tiempo real")
        st.info(
            "Permite el acceso a tu cámara para detectar enfermedades en tiempo real."
        )

        webrtc_streamer(
            key="deteccion-plantas",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIG,  # ¡Esta línea soluciona el error!
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

if __name__ == "__main__":
    main()
