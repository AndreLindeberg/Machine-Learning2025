import streamlit as st
import joblib
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
from skimage.filters import threshold_otsu 

# Ladda den sparade modellen
model = joblib.load("best_xgb_model.pkl")
scaler = joblib.load("scaler.pkl") # Om den behövs

# Funktion för att binarisera.
def binarize_image(image_array):
    threshold_value = threshold_otsu(image_array)
    image_bin = image_array > threshold_value
    return image_bin.astype(np.uint8) * 255

def preprocess_image(image_array):
    # Konvertera till gråskala
    image = Image.fromarray(image_array.astype("uint8")).convert('L')
    img_array = np.array(image)

    # Otsus tröskling
    _, img_bin = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Säkerställer att bakgrunden är svart och siffran är vit
    if np.mean(img_bin) > 127:
        img_bin = cv2.bitwise_not(img_bin)  # Invertera om nödvändigt

    # Hitta konturer och centrera siffran
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(contours[0])
        digit = img_bin[y:y+h, x:x+w]

        # Anpassa storlek utan att förvränga proportionerna
        new_w = 20 if w > h else int((w / h) * 20)
        new_h = 20 if h > w else int((h / w) * 20)
        digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Skapa en tom 28x28 bild med helt svart bakgrund
        img_padded = np.zeros((28, 28), dtype=np.uint8)

        # Räkna ut var den beskurna siffran ska placeras i mitten
        x_offset, y_offset = (28 - new_w) // 2, (28 - new_h) // 2

        # Placera den förstorade siffran i mitten av 28x28-bilden
        img_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

    else:
        img_padded = cv2.resize(img_bin, (28, 28))

    # Centrera siffran med warpAffine
    moments = cv2.moments(img_padded)
    if moments["m00"] != 0:
        cX, cY = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
        M = np.float32([[1, 0, 14 - cX], [0, 1, 14 - cY]])
        img_padded = cv2.warpAffine(img_padded, M, (28, 28), borderValue=0)

    # Visa bilden i Streamlit
    st.image(img_padded, caption="Förbehandlad bild", width=150)

    # Normalisering & Standardisering
    img_padded = img_padded.reshape(1, -1)  # Gör om till vektor (1, 784)
 
    return img_padded


# Skapa Streamlit-app.
st.title("✍️ MNIST Sifferklassificering med XGBoost")
st.write("Rita en siffra i rutan nedan och låt modellen gissa!")

# Skapa canvas.
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=20,
    stroke_color="white",
    background_color="#000000",
    height=280, width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Kontrollera att användaren har ritat något.
if canvas_result.image_data is not None:
    
    image_array = np.array(canvas_result.image_data[:, :, :3], dtype=np.uint8) # Hämta bilddata från canvas
    if np.count_nonzero(image_array) < 10:  # Mindre än 10 pixlar ≈ tom canvas
        st.warning("Ingen siffra ritad! Rita en siffra innan du gör en prediktion.")
        st.stop()

    #  Förbehandla bilden
    processed_img = preprocess_image(image_array)
    
    probs = model.predict_proba(processed_img) # 🔎 Se vad modellen tror
    top_prediction = np.argmax(probs) #  Är modellen säker nog?

    if probs[0, top_prediction] < 0.5:  # Om sannolikheten är låg
        st.warning("Osäker prediktion! Modellen kan vara tveksam.")
    else:
        st.success(f"🧠 Modellen tror att detta är en **{top_prediction}**!")
