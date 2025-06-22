# Fruit & Vegetable Name Classification - CNN

This project is a web application that classifies images of fruits and vegetables using a Convolutional Neural Network (CNN) model.

## Project Structure

```
Fruits_Vegetables_model_96.10.h5   # Trained Keras model
images.jpg                         # Example image
Model.ipynb                        # Jupyter notebook for model training
Flask Website/
    app.py                         # Flask web server
    templates/
        index.html                 # Web interface
```

## Setup Instructions

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Run the Flask app:**
   ```
   python app.py
   ```

3. **Open your browser and go to:**
   ```
   http://127.0.0.1:5000/
   ```

## Files

- `Fruits_Vegetables_model_96.10.h5`: Pre-trained CNN model.
- `Model.ipynb`: Notebook for model training and evaluation.
- `Flask Website/app.py`: Main Flask application.
- `Flask Website/templates/index.html`: HTML template for the web interface.

## Usage

Upload an image of a fruit or vegetable using the web interface. The app will predict and display the name of the item.
