from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import base64

app = Flask(__name__)

model = load_model("C:\\Users\\Abdullah Tariq\\Desktop\\collage project\\Fruits_Vegetables_model_96.10.h5")


CLASS_LABELS = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

def preprocess_image(image):
    """Preprocess image to match model input shape."""
    image = cv2.resize(image, (128, 128))       
    image = image.astype('float32') / 255.0    
    image = np.expand_dims(image, axis=0)     
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_data = None
    if request.method == "POST":
        if "file" not in request.files:
            result = "No file uploaded"
        else:
            file = request.files["file"]
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                result = "Could not read the image"
            else:
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)
                class_index = np.argmax(predictions)  
                class_name = CLASS_LABELS[class_index]
                result = f"This is a {class_name}"

                
                success, buffer = cv2.imencode('.jpg', image)
                if success:
                    img_data = base64.b64encode(buffer).decode('utf-8')
                else:
                    result = "Error encoding image for display"

    return render_template("index.html", result=result, img_data=img_data)

if __name__ == "__main__":
    app.run(debug=True)
