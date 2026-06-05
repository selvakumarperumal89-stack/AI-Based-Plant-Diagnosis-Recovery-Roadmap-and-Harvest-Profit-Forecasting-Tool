from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import pandas as pd
import os
from ai_edge_litert import interpreter as tflite

app = Flask(__name__)

base_dir = os.path.join(os.path.dirname(__file__), "models", "tflite_models")

model_paths = [
    os.path.join(base_dir, "inceptionv3_model.tflite"),
    os.path.join(base_dir, "resnet101_model.tflite"),
    os.path.join(base_dir, "vgg16_model.tflite"),
]

meta_path = os.path.join(os.path.dirname(__file__), "models", "metadata.csv")

DIAGNOSES = {
    "Chili__healthy": "Your chili plant is healthy. No disease detected.",
    "Chili__leaf curl": "Chili leaf curl is caused by viral infection or mite infestation.",
    "Chili__leaf spot": "Chili leaf spot is caused by fungal pathogens affecting the leaves.",
    "Chili__whitefly": "Whitefly infestation detected on chili plant.",
    "Chili__yellowish": "Yellowing of chili leaves indicates nutrient deficiency or disease.",
    "Corn__Common_rust": "Common rust in corn is caused by Puccinia sorghi fungus.",
    "Corn__Gray_Leaf_Spot": "Gray leaf spot in corn is caused by Cercospora zeae-maydis.",
    "Corn__Healthy": "Your corn plant is healthy. No disease detected.",
    "Corn__Northern_Leaf_Blight": "Northern leaf blight is caused by Exserohilum turcicum.",
    "Potato__Early_blight": "Early blight in potato is caused by Alternaria solani.",
    "Potato__Healthy": "Your potato plant is healthy. No disease detected.",
    "Potato__Late_blight": "Late blight in potato is caused by Phytophthora infestans.",
    "Sugarcane__Bacterial Blight": "Bacterial blight in sugarcane is caused by Xanthomonas albilineans.",
    "Sugarcane__Healthy": "Your sugarcane plant is healthy. No disease detected.",
    "Sugarcane__Red Rot": "Red rot in sugarcane is caused by Colletotrichum falcatum.",
    "Sugarcane__Red_rust": "Red rust in sugarcane is caused by Phakopsora pachyrhizi.",
    "Sugarcane__Yellow": "Yellowing in sugarcane may indicate nutrient deficiency or disease.",
    "Tomato__Bacterial_spot": "Bacterial spot in tomato is caused by Xanthomonas campestris.",
    "Tomato__Early_blight": "Early blight in tomato is caused by Alternaria solani.",
    "Tomato__Healthy": "Your tomato plant is healthy. No disease detected.",
    "Tomato__Late_blight": "Late blight in tomato is caused by Phytophthora infestans.",
    "Tomato__Leaf_Mold": "Leaf mold in tomato is caused by Passalora fulva.",
    "Tomato__Mosaic_virus": "Tomato mosaic virus is a viral disease spread by contact.",
    "Tomato__Septoria_leaf_spot": "Septoria leaf spot is caused by Septoria lycopersici.",
    "Tomato__Spider_mites": "Spider mite infestation detected on tomato plant.",
    "Tomato__Target_Spot": "Target spot in tomato is caused by Corynespora cassiicola.",
    "Tomato__Yellow_Leaf_Curl_Virus": "Yellow leaf curl virus is spread by whiteflies.",
    "Tomato__powdery_mildew": "Powdery mildew in tomato is caused by Oidium neolycopersici.",
}

RECOVERY_STEPS = {
    "Chili__healthy": "No treatment needed. Maintain regular watering and fertilization.",
    "Chili__leaf curl": "1. Remove infected leaves.\n2. Apply neem oil spray.\n3. Use systemic insecticide for mites.\n4. Ensure proper irrigation.",
    "Chili__leaf spot": "1. Remove infected leaves.\n2. Apply copper-based fungicide.\n3. Avoid overhead irrigation.\n4. Improve air circulation.",
    "Chili__whitefly": "1. Use yellow sticky traps.\n2. Apply insecticidal soap.\n3. Introduce natural predators.\n4. Apply neem oil spray.",
    "Chili__yellowish": "1. Test soil nutrients.\n2. Apply balanced NPK fertilizer.\n3. Ensure proper irrigation.\n4. Check for root diseases.",
    "Corn__Common_rust": "1. Apply fungicide at early stages.\n2. Use resistant varieties.\n3. Remove infected plant debris.\n4. Ensure proper spacing.",
    "Corn__Gray_Leaf_Spot": "1. Apply fungicide.\n2. Rotate crops.\n3. Remove infected debris.\n4. Use resistant hybrids.",
    "Corn__Healthy": "No treatment needed. Maintain regular care.",
    "Corn__Northern_Leaf_Blight": "1. Apply fungicide.\n2. Use resistant varieties.\n3. Crop rotation.\n4. Remove crop debris.",
    "Potato__Early_blight": "1. Apply fungicide every 7-10 days.\n2. Remove infected leaves.\n3. Avoid overhead irrigation.\n4. Improve drainage.",
    "Potato__Healthy": "No treatment needed. Maintain regular care.",
    "Potato__Late_blight": "1. Apply copper-based fungicide.\n2. Remove infected plants.\n3. Avoid excess moisture.\n4. Use resistant varieties.",
    "Sugarcane__Bacterial Blight": "1. Remove and destroy infected plants.\n2. Use disease-free planting material.\n3. Apply copper bactericide.\n4. Improve drainage.",
    "Sugarcane__Healthy": "No treatment needed. Maintain regular care.",
    "Sugarcane__Red Rot": "1. Remove infected stalks.\n2. Treat seed cane with fungicide.\n3. Use resistant varieties.\n4. Improve field drainage.",
    "Sugarcane__Red_rust": "1. Apply fungicide spray.\n2. Remove infected leaves.\n3. Improve air circulation.\n4. Use resistant varieties.",
    "Sugarcane__Yellow": "1. Test soil nutrients.\n2. Apply nitrogen fertilizer.\n3. Check irrigation levels.\n4. Monitor for virus infections.",
    "Tomato__Bacterial_spot": "1. Apply copper-based bactericide.\n2. Remove infected leaves.\n3. Avoid overhead watering.\n4. Use disease-free seeds.",
    "Tomato__Early_blight": "1. Apply fungicide.\n2. Remove lower infected leaves.\n3. Mulch around plants.\n4. Ensure proper spacing.",
    "Tomato__Healthy": "No treatment needed. Maintain regular care.",
    "Tomato__Late_blight": "1. Apply copper fungicide.\n2. Remove infected parts.\n3. Avoid wet foliage.\n4. Improve air circulation.",
    "Tomato__Leaf_Mold": "1. Apply fungicide.\n2. Improve ventilation.\n3. Reduce humidity.\n4. Remove infected leaves.",
    "Tomato__Mosaic_virus": "1. Remove and destroy infected plants.\n2. Control aphid vectors.\n3. Disinfect tools.\n4. Use resistant varieties.",
    "Tomato__Septoria_leaf_spot": "1. Apply fungicide.\n2. Remove infected leaves.\n3. Avoid wetting leaves.\n4. Rotate crops.",
    "Tomato__Spider_mites": "1. Apply miticide or neem oil.\n2. Increase humidity.\n3. Remove heavily infested leaves.\n4. Introduce predatory mites.",
    "Tomato__Target_Spot": "1. Apply fungicide.\n2. Remove infected leaves.\n3. Improve air circulation.\n4. Avoid overhead irrigation.",
    "Tomato__Yellow_Leaf_Curl_Virus": "1. Control whitefly population.\n2. Remove infected plants.\n3. Use reflective mulches.\n4. Use resistant varieties.",
    "Tomato__powdery_mildew": "1. Apply sulfur-based fungicide.\n2. Remove infected leaves.\n3. Improve air circulation.\n4. Avoid excess nitrogen.",
}

YIELD_IMPACT = {
    "Chili__healthy": 0,
    "Chili__leaf curl": 40,
    "Chili__leaf spot": 25,
    "Chili__whitefly": 30,
    "Chili__yellowish": 20,
    "Corn__Common_rust": 30,
    "Corn__Gray_Leaf_Spot": 35,
    "Corn__Healthy": 0,
    "Corn__Northern_Leaf_Blight": 40,
    "Potato__Early_blight": 30,
    "Potato__Healthy": 0,
    "Potato__Late_blight": 50,
    "Sugarcane__Bacterial Blight": 40,
    "Sugarcane__Healthy": 0,
    "Sugarcane__Red Rot": 45,
    "Sugarcane__Red_rust": 25,
    "Sugarcane__Yellow": 20,
    "Tomato__Bacterial_spot": 30,
    "Tomato__Early_blight": 25,
    "Tomato__Healthy": 0,
    "Tomato__Late_blight": 50,
    "Tomato__Leaf_Mold": 20,
    "Tomato__Mosaic_virus": 35,
    "Tomato__Septoria_leaf_spot": 25,
    "Tomato__Spider_mites": 20,
    "Tomato__Target_Spot": 25,
    "Tomato__Yellow_Leaf_Curl_Virus": 40,
    "Tomato__powdery_mildew": 20,
}


def load_model_dynamic(path):
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def run_model(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        input_data = preprocess_image(image)

        df = pd.read_csv(meta_path)
        class_names = df['class_name'].tolist()

        all_preds = []
        for path in model_paths:
            if os.path.exists(path):
                interp = load_model_dynamic(path)
                preds = run_model(interp, input_data)
                all_preds.append(preds)

        if not all_preds:
            return jsonify({'error': 'No models found'}), 500

        avg_preds = np.mean(all_preds, axis=0)
        predicted_index = int(np.argmax(avg_preds))
        confidence = float(np.max(avg_preds)) * 100
        predicted_class = class_names[predicted_index]

        diagnosis = DIAGNOSES.get(predicted_class, "Unknown disease detected.")
        recovery = RECOVERY_STEPS.get(predicted_class, "Consult a local agricultural expert.")
        yield_loss = YIELD_IMPACT.get(predicted_class, 0)

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
    # For mobile app connection
    app.run(host='0.0.0.0', port=8081, debug=True)
