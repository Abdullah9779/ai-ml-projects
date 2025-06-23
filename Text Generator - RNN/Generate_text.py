import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model

with open("tokenizer.pickle", "rb") as file:
  tokenizer = pickle.load(file)

model = load_model("Text Generator.h5")
y_array = pd.read_csv("y.csv")
y_array = y_array["array"].to_numpy()

def genrate_text(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=147, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    index = np.argmax(predicted)
    return tokenizer.index_word[y_array[index]]

text = "The python"

for _ in range(30):
  word = genrate_text(text)
  text += " " + word

print("\n\n-->", text)
