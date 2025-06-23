# Text Generator - RNN

This project is a Recurrent Neural Network (RNN) based text generator implemented in Python. It uses Keras and TensorFlow to train a model on a text dataset and generate new text sequences.

## Project Structure
- `Generate_text.py`: Script to generate text using the trained RNN model.
- `Text Generator.ipynb`: Jupyter notebook for training and experimenting with the RNN model.
- `Text Generator.h5`: Saved Keras model weights.
- `tokenizer.pickle`: Tokenizer object used for text preprocessing.
- `Data.txt`: Input text data used for training.
- `y.csv`: Additional data, possibly labels or processed sequences.

## Usage
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the model:**
   Use the Jupyter notebook `Text Generator.ipynb` to train the model on your dataset.
3. **Generate text:**
   Run the script:
   ```bash
   python Generate_text.py
   ```

## Requirements
See `requirements.txt` for the list of dependencies.

## Notes
- Make sure you have the required data files (`Data.txt`, `tokenizer.pickle`, etc.) in the project directory.
- The model and tokenizer are saved after training and loaded for text generation.
