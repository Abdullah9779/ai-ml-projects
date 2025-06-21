import json
import numpy as np
import pickle
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

model = load_model("chatbot.h5")

with open("index_tag.json", "rb") as file:
    index_tag = json.load(file)

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("intents.json") as file:
    data = json.load(file)

def get_prediction(text):
    sequences = tokenizer.texts_to_sequences([text])
    sequences = pad_sequences(sequences, maxlen=8, padding="post")
    sequences = np.array(sequences)
    pred = model.predict(sequences)
    index = np.argmax(pred)
    print(pred[0][index])
    return index_tag[str(index)] if pred[0][index] > 0.25 else None
    
def get_answer(tag):
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
        
        
def chat():
    while True:
        user_input = input("You : ")
        if user_input in ["quit", "exit"]:
            break
        
        tag = get_prediction(user_input)
        print(tag)
        if tag is not None:
            answer = get_answer(tag)
            print(f"Bot : {answer}")
        else:
            print("Bot : Sorry i cannot understand")
            
if __name__ == "__main__":
    chat()
