from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir peticiones desde tu app React Native

# Cargar el modelo entrenado
model = load_model("modelo_flores.h5")

# Etiquetas de clases según el entrenamiento
labels = ['girasol', 'magnolia', 'orquidea']

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
        print(f"Procesando archivo: {file.filename}")  # Log del archivo recibido
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        pred = model.predict(img_array)
        class_index = np.argmax(pred)
        predicted_label = labels[class_index]

        print(f"Predicción: {predicted_label}")  # Log de la predicción
        return jsonify({'prediction': predicted_label})
    except Exception as e:
        print(f"Error procesando la imagen: {str(e)}")  # Log del error
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
