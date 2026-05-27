# =========================
# IMPORTS
# =========================
import streamlit as st
import cv2
import tempfile
from PIL import Image
import tensorflow as tf
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

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

def inject_custom_css():
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(180deg, #f4fbf5 0%, #ffffff 100%);
        }

        .hero {
            background: linear-gradient(135deg, #14532d, #2e7d32, #66bb6a);
            padding: 2.2rem;
            border-radius: 24px;
            color: white;
            box-shadow: 0 12px 32px rgba(20, 83, 45, 0.25);
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.4rem;
            font-weight: 800;
        }

        .hero p {
            margin-top: .6rem;
            font-size: 1.05rem;
            opacity: .95;
        }

        .card {
            background: white;
            color: #0f172a;
            padding: 1.3rem;
            border-radius: 20px;
            border: 1px solid #dbeafe;
            box-shadow: 0 8px 24px rgba(0,0,0,.07);
            margin-bottom: 1rem;
        }

        .result-ok {
            background: linear-gradient(135deg, #dcfce7, #ffffff);
            border-left: 8px solid #16a34a;
        }

        .result-bad {
            background: linear-gradient(135deg, #fee2e2, #ffffff);
            border-left: 8px solid #dc2626;
        }

        .result-title {
            font-size: 1.7rem;
            font-weight: 800;
            margin-bottom: .4rem;
            color: #0f172a;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #14532d;
            margin: 1rem 0 .6rem 0;
        }

        .muted {
            color: #475569;
            font-size: .95rem;
        }
    </style>
    """, unsafe_allow_html=True)


def render_result_card(estado, clase, confidence, key):
    css_class = "result-ok" if estado == "Sana" else "result-bad"
    icon = "✅" if estado == "Sana" else "⚠️"

    st.markdown(f"""
    <div class="card {css_class}">
        <div class="result-title">{icon} {estado}</div>
        <p><b>Clase detectada:</b> {clase}</p>
        <p><b>Confianza del modelo:</b> {confidence:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    if key in info_enfermedades:
        st.markdown("""
        <div class="section-title">📖 Información agronómica</div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="card">
                <h4>🌿 Planta</h4>
                <p>{info_enfermedades[key]['planta']}</p>
                <h4>🧬 Descripción</h4>
                <p>{info_enfermedades[key]['descripcion']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card">
                <h4>🛠️ Recomendación</h4>
                <p>{info_enfermedades[key]['recomendacion']}</p>
            </div>
            """, unsafe_allow_html=True)

# =========================
# CARGA DEL MODELO
# =========================
@st.cache_resource
def load_model():
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
# VIDEO
# =========================
def process_video(video_file, model):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    sanas, enfermas = 0, 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue

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

        h, w, _ = frame.shape
        cv2.rectangle(frame, (10, 10), (w-10, h-10), color, 3)
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()

    st.subheader("📊 Resumen del video")
    col1, col2 = st.columns(2)
    col1.metric("Frames sanos", sanas)
    col2.metric("Frames enfermos", enfermas)

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
    inject_custom_css()

    st.markdown("""
    <div class="hero">
        <h1>🌱 Sistema Inteligente de Detección de Enfermedades en Plantas</h1>
        <p>Deep Learning aplicado a la agricultura para el análisis de hojas mediante imagen, video y cámara en tiempo real.</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.title("📋 Panel de Control")

    opcion = st.sidebar.radio(
        "Modo de análisis",
        ("Imagen", "Video", "Cámara en tiempo real")
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Modelo")
    st.sidebar.info("Clasificador de enfermedades foliares basado en redes neuronales profundas.")

    st.sidebar.markdown("### 🌿 Cultivos soportados")
    st.sidebar.write(
        "Manzana, Maíz, Uva, Tomate, Papa, Fresa, Cítricos, Pimiento y otros cultivos del dataset."
    )

    st.sidebar.markdown("### ⚠️ Nota")
    st.sidebar.caption(
        "El resultado es una predicción orientativa. Para decisiones agrícolas reales, se recomienda validación de un especialista."
    )

    model = load_model()

    if opcion == "Imagen":
        st.markdown('<div class="section-title">📷 Análisis por imagen</div>', unsafe_allow_html=True)

        col_img, col_result = st.columns([1.05, 1.25], gap="large")

        with col_img:
            file = st.file_uploader(
                "Carga una imagen de una hoja",
                type=["jpg", "png", "jpeg"]
            )

            if file:
                image = Image.open(file).convert("RGB")
                st.image(image, caption="Imagen cargada", width="stretch")
            else:
                st.markdown("""
                <div class="card">
                    <h4>Instrucciones</h4>
                    <p>Sube una imagen clara de una hoja. Para mejores resultados, usa buena iluminación y evita fondos muy cargados.</p>
                </div>
                """, unsafe_allow_html=True)

        with col_result:
            st.markdown('<div class="section-title">🩺 Resultado del diagnóstico</div>', unsafe_allow_html=True)

            if file:
                img = process_image_pil(image)
                pred = model.predict(img, verbose=0)
                label = np.argmax(pred)
                confidence = np.max(pred) * 100

                key = labels[label]
                estado = "Sana" if "healthy" in key else "Enferma"
                clase = traducciones[label]

                col1, col2, col3 = st.columns(3)
                col1.metric("Estado", estado)
                col2.metric("Confianza", f"{confidence:.2f}%")
                col3.metric("Clase", clase)

                render_result_card(estado, clase, confidence, key)
            else:
                st.markdown("""
                <div class="card">
                    <h4>Esperando imagen...</h4>
                    <p>Cuando cargues una imagen, el diagnóstico aparecerá en esta sección.</p>
                </div>
                """, unsafe_allow_html=True)

    elif opcion == "Video":
        st.markdown('<div class="section-title">🎥 Análisis por video</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <p>Sube un video corto donde la hoja aparezca visible. El sistema analizará frames del video y mostrará si la hoja se encuentra sana o enferma.</p>
        </div>
        """, unsafe_allow_html=True)

        video = st.file_uploader(
            "Carga un video",
            type=["mp4", "avi", "mov"]
        )

        if video:
            st.info("Procesando video...")
            process_video(video, model)

    elif opcion == "Cámara en tiempo real":
        st.markdown('<div class="section-title">🎦 Detección en tiempo real</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <p>Permite el acceso a tu cámara y coloca la hoja frente al lente. El sistema mostrará el estado de la hoja en tiempo real.</p>
        </div>
        """, unsafe_allow_html=True)

        st.warning(
            "En celulares, la cámara en tiempo real puede requerir HTTPS. Si falla, usa el modo Imagen."
        )

        webrtc_streamer(
            key="deteccion-plantas",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

if __name__ == "__main__":
    main()
