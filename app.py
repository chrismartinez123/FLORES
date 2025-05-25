# 1. Imports
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import numpy as np
import os
import requests

# 2. Configuración de Flask y CORS
app = Flask(__name__)
CORS(app)

# 3. Configuración de ruta del modelo y URL de Google Drive
MODEL_PATH = "modelo_flores.h5"
FILE_ID = "1s9aKHSb9w_FjWnq285hNEpCWT886bEr7"  # Solo el ID del archivo

# 4. Función para descargar archivo grande de Google Drive con manejo de confirmación
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# 5. Descargar modelo si no existe
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo desde Google Drive...")
        download_file_from_google_drive(FILE_ID, MODEL_PATH)
        print("Modelo descargado correctamente.")
    else:
        print("El modelo ya está descargado.")

# 6. Llamar descarga y cargar modelo
download_model()
model = load_model(MODEL_PATH)

# 7. Etiquetas de clases
labels = ['girasol', 'magnolia', 'orquidea']

# 8. Rutas Flask
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

# 9. Ejecutar app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

