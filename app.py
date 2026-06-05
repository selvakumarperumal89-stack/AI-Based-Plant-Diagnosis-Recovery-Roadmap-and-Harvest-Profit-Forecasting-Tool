from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import os
import urllib.request
import json

app = Flask(__name__)

# Model will be downloaded automatically from internet
MODEL_URL = "https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_100_224/classification/5/metadata/1?lite-format=tflite"
MODEL_PATH = "/tmp/plant_model.tflite"

# Plant Village class names (38 classes)
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

DIAGNOSES = {
    "Potato___Early_blight": "Early blight in potato is caused by Alternaria solani fungus.",
    "Potato___Late_blight": "Late blight in potato is caused by Phytophthora infestans.",
    "Potato___healthy": "Your potato plant is healthy. No disease detected.",
    "Corn_(maize)___Common_rust_": "Common rust in corn is caused by Puccinia sorghi fungus.",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern leaf blight is caused by Exserohilum turcicum.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gray leaf spot in corn is caused by Cercospora zeae-maydis.",
    "Corn_(maize)___healthy": "Your corn plant is healthy. No disease detected.",
    "Tomato___Bacterial_spot": "Bacterial spot in tomato is caused by Xanthomonas campestris.",
    "Tomato___Early_blight": "Early blight in tomato is caused by Alternaria solani.",
    "Tomato___Late_blight": "Late blight in tomato is caused by Phytophthora infestans.",
    "Tomato___Leaf_Mold": "Leaf mold in tomato is caused by Passalora fulva.",
    "Tomato___Septoria_leaf_spot": "Septoria leaf spot is caused by Septoria lycopersici.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider mite infestation detected on tomato plant.",
    "Tomato___Target_Spot": "Target spot in tomato is caused by Corynespora cassiicola.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow leaf curl virus is spread by whiteflies.",
    "Tomato___Tomato_mosaic_virus": "Tomato mosaic virus is a viral disease spread by contact.",
    "Tomato___healthy": "Your tomato plant is healthy. No disease detected.",
}

RECOVERY_STEPS = {
    "Potato___Early_blight": "1. Apply fungicide every 7-10 days.\n2. Remove infected leaves.\n3. Avoid overhead irrigation.\n4. Improve drainage.",
    "Potato___Late_blight": "1. Apply copper-based fungicide.\n2. Remove infected plants.\n3. Avoid excess moisture.\n4. Use resistant varieties.",
    "Potato___healthy": "No treatment needed. Maintain regular care.",
    "Corn_(maize)___Common_rust_": "1. Apply fungicide at early stages.\n2. Use resistant varieties.\n3. Remove infected plant debris.\n4. Ensure proper spacing.",
    "Corn_(maize)___Northern_Leaf_Blight": "1. Apply fungicide.\n2. Use resistant varieties.\n3. Crop rotation.\n4. Remove crop debris.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "1. Apply fungicide.\n2. Rotate crops.\n3. Remove infected debris.\n4. Use resistant hybrids.",
    "Corn_(maize)___healthy": "No treatment needed. Maintain regular care.",
    "Tomato___Bacterial_spot": "1. Apply copper-based bactericide.\n2. Remove infected leaves.\n3. Avoid overhead watering.\n4. Use disease-free seeds.",
    "Tomato___Early_blight": "1. Apply fungicide.\n2. Remove lower infected leaves.\n3. Mulch around plants.\n4. Ensure proper spacing.",
    "Tomato___Late_blight": "1. Apply copper fungicide.\n2. Remove infected parts.\n3. Avoid wet foliage.\n4. Improve air circulation.",
    "Tomato___Leaf_Mold": "1. Apply fungicide.\n2. Improve ventilation.\n3. Reduce humidity.\n4. Remove infected leaves.",
    "Tomato___Septoria_leaf_spot": "1. Apply fungicide.\n2. Remove infected leaves.\n3. Avoid wetting leaves.\n4. Rotate crops.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "1. Apply miticide or neem oil.\n2. Increase humidity.\n3. Remove heavily infested leaves.\n4. Introduce predatory mites.",
    "Tomato___Target_Spot": "1. Apply fungicide.\n2. Remove infected leaves.\n3. Improve air circulation.\n4. Avoid overhead irrigation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "1. Control whitefly population.\n2. Remove infected plants.\n3. Use reflective mulches.\n4. Use resistant varieties.",
    "Tomato___Tomato_mosaic_virus": "1. Remove and destroy infected plants.\n2. Control aphid vectors.\n3. Disinfect tools.\n4. Use resistant varieties.",
    "Tomato___healthy": "No treatment needed. Maintain regular care.",
}

YIELD_IMPACT = {
    "Potato___Early_blight": 30, "Potato___Late_blight": 50, "Potato___healthy": 0,
    "Corn_(maize)___Common_rust_": 30, "Corn_(maize)___Northern_Leaf_Blight": 40,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 35, "Corn_(maize)___healthy": 0,
    "Tomato___Bacterial_spot": 30, "Tomato___Early_blight": 25, "Tomato___Late_blight": 50,
    "Tomato___Leaf_Mold": 20, "Tomato___Septoria_leaf_spot": 25,
    "Tomato___Spider_mites Two-spotted_spider_mite": 20, "Tomato___Target_Spot": 25,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 40, "Tomato___Tomato_mosaic_virus": 35,
    "Tomato___healthy": 0,
}

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return jsonify({'status': 'AI Plant Diagnosis API is running!', 'endpoint': '/predict'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        download_model()

        from ai_edge_litert import interpreter as tflite
        interp = tflite.Interpreter(model_path=MODEL_PATH)
        interp.allocate_tensors()

        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        input_data = preprocess_image(image)

        input_details = interp.get_input_details()
        output_details = interp.get_output_details()
        interp.set_tensor(input_details[0]['index'], input_data)
        interp.invoke()
        output = interp.get_tensor(output_details[0]['index'])[0]

        predicted_index = int(np.argmax(output))
        confidence = float(np.max(output)) * 100

        if predicted_index < len(CLASS_NAMES):
            predicted_class = CLASS_NAMES[predicted_index]
        else:
            predicted_class = "Unknown"

        diagnosis = DIAGNOSES.get(predicted_class, "Disease detected. Please consult a local agricultural expert.")
        recovery = RECOVERY_STEPS.get(predicted_class, "Consult a local agricultural expert for treatment advice.")
        yield_loss = YIELD_IMPACT.get(predicted_class, 20)

        return jsonify({
            'disease': predicted_class,
            'confidence': round(confidence, 2),
            'diagnosis': diagnosis,
            'recovery_steps': recovery,
            'yield_impact_percent': yield_loss
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)
