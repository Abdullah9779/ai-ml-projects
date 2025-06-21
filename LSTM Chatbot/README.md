# Chatbot Project

This is a simple chatbot project using TensorFlow and Keras. It uses an LSTM-based neural network to classify user input into intents and generate responses.

## Features
- Intent classification using LSTM
- Tokenization and sequence padding
- Model training and saving
- Prediction on new user input

## Files
- `main.py`: Script for loading the model and making predictions.
- `Chatbot.ipynb`: Jupyter notebook for data preprocessing, model training, and saving artifacts.
- `Chatbot.h5`: Trained Keras model.
- `tokenizer.pkl`: Saved tokenizer for text preprocessing.
- `index_tag.json`: Mapping of class indices to intent tags.
- `intents.json`: Dataset of intents and patterns.

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the notebook to train the model or use `main.py` to make predictions.

## Usage
- To train the model, open and run all cells in `Chatbot.ipynb`.
- To make predictions, run `main.py` after training and saving the model.

## Requirements
- Python 3.7+
- TensorFlow
- NumPy
- scikit-learn

