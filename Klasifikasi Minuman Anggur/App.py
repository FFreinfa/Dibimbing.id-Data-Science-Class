from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model dan scaler
with open("model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

# Peta kelas anggur ke gambar
wine_images = {
    0: "/static/red wine.png",
    1: "/static/White Wine.webp",
    2: "/static/Rose Wine.jpg"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data dari form HTML
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)

        # Standarisasi fitur
        features_scaled = scaler.transform(features)

        # Prediksi kelas anggur
        prediction = model.predict(features_scaled)[0]
        wine_classes = ["Kelas 0 - Anggur Merah", "Kelas 1 - Anggur Putih", "Kelas 2 - Anggur Rosé"]
        result = wine_classes[prediction]

        # Gambar sesuai prediksi
        image_url = wine_images[prediction]

        return render_template("Hasil.html", result=result, image_url=image_url)
    
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
