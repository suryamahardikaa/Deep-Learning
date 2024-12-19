from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Variabel
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model h5
model_path = "FishFreshness_2.h5"
model = tf.keras.models.load_model(model_path)

# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi prediksi
def predict(model, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(250, 250))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)

    # Menentukan kategori berdasarkan nilai prediksi
    if prediction < 0.5:
        return {"result": "Ikan termasuk dalam kategori segar."}
    else:
        return {"result": "Ikan termasuk dalam kategori tidak segar."}

# @app.route('/')
# def index():
#     return render_template('index.html', message=None)

@app.route('/predict', methods=['POST'])
def predict_image():
    # Memeriksa apakah ada file yang diunggah
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "Tidak ada gambar ikan yang dimasukkan."}), 400

    # Memeriksa apakah file yang diunggah valid
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Lakukan prediksi
            result = predict(model, filepath)
            response = {
                "message": "Success Predict",
                "result": result["result"],
            }
            return jsonify(response)

        except Exception as e:
            return jsonify({"message": f"Terjadi kesalahan saat melakukan prediksi"}), 500

    else:
        return jsonify({"message": "File yang diunggah tidak valid. Harap unggah gambar dengan format JPG, JPEG, atau PNG."}), 400

if __name__ == '__main__':
    app.run(debug=True)
