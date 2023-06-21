import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import dump, load
from keras.models import Model, load_model


def get_features(image, feature_model):
    image = cv2.resize(image,(299,299))
    #print(image[10,10])

    image = np.array(image)
    
    # convert images into three channels just in case if its 4 channel image
    #image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = feature_model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    caption = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        caption += ' ' + word
        if word == 'end':
            break
    return caption

def main():
    img_path = input('Enter image path: ')

    while img_path == '':
        img_path = input('Enter image path: ')

    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        max_length = 32
        tokenizer = load(open("tokenizer.p","rb"))
        model = load_model('models/model_9.h5')
        xception_model = Xception(include_top=False, pooling="avg")
        photo = get_features(img, xception_model)
        caption = generate_desc(model, tokenizer, photo, max_length)
        print(caption[:-4])
        blank_image = img.copy()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imshow("Image",img)
        cv2.waitKey(0)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()

