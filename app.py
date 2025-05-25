# 1. Imports
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import numpy as np
import os
import requests  # ← necesario para descargar desde Google Drive

# 2. Configuración de Flask y CORS
app = Flask(__name__)
CORS(app)

# 3. Configuración de ruta del modelo y URL de Google Drive
MODEL_PATH = "modelo_flores.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1s9aKHSb9w_FjWnq285hNEpCWT886bEr7"

# 4. Función para descargar el modelo si no existe
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo desde Google Drive...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Modelo descargado correctamente.")
    else:
        print("El modelo ya está descargado.")

# 5. Llamar la función de descarga ANTES de cargar el modelo
download_model()

# 6. Cargar el modelo desde archivo
model = load_model(MODEL_PATH)

# 7. Etiquetas de clases
labels = ['girasol', 'magnolia', 'orquidea']

# 8. Rutas de Flask (home y predict)
@app.route('/')
def home():
    return "API funcionando correctamente"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Archivo no encontrado'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'Archivo inválido. Solo se permiten imágenes JPG o PNG.'}), 400

    os.makedirs('temp', exist_ok=True)
    img_path = os.path.join('temp', file.filename)
    file.save(img_path)

    try:
        print(f"Procesando archivo: {file.filename}")
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        pred = model.predict(img_array)
        class_index = np.argmax(pred)
        predicted_label = labels[class_index]

        return jsonify({'prediction': predicted_label})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

# 9. Iniciar la app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

