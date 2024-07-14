import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import cv2
from ultralytics import YOLO

data = pd.read_csv('FINAL_COMBINED_DATASET_3CAP.csv')

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data["urdu_caption"])
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(ucap.split()) for ucap in data['urdu_caption'])

def idx_to_word(integer, tokenizer):
    for word,index, in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'endseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        
        # get index with high probability
        yhat = np.argmax(yhat)
        
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        
        # stop if we reach end tag
        if word == 'startseq':
            break
    return in_text

cnn = YOLO("yolov8m-cls.pt")

def generate_caption(image_name, model):
    image = Image.open(image_name)
    features = cnn.predict(image, embed = [-1])

    # Predict caption for image
    y_pred = predict_caption(model, features, tokenizer, max_length)[8:][:-6]
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)


def get_caption_model():
    m = load_model('Trained_YOLO_LSTM_3c2e.h5')
    return m

