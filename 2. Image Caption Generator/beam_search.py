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


def generate_desc(model, tokenizer, photo, max_length, beam_index=3):
    # Start with the initial token 'start'
    in_text = 'start'
    
    # Initialize the sequence with the start token
    seq = [in_text]
    
    # Initialize the beam search with the first word
    start = [[in_text, 0.0]]
    
    # Keep looping until maximum length is reached
    for i in range(max_length):
        # Initialize a new list to store the candidates for the next step
        candidates = []
        
        # Loop through each sentence in the current beam
        for s in start:
            # Get the current sentence and its score
            seq = s[0]
            score = s[1]
            
            # If the current sentence ends with 'end', add it to the candidates
            if seq.split()[-1] == 'end':
                candidates.append(s)
                continue
            
            # Get the current sequence of word indices
            sequence = tokenizer.texts_to_sequences([seq])[0]
            
            # Pad the sequence to the maximum length
            sequence = pad_sequences([sequence], maxlen=max_length)
            
            # Predict the class probabilities for the next word
            yhat = model.predict([photo, sequence], verbose=0)
            
            # Get the indices of the top k predictions using beam search
            top_word_indices = yhat.argsort()[0][-beam_index:]
            
            # Loop through each of the top k predictions
            for j in top_word_indices:
                # Get the corresponding word
                word = word_for_id(j, tokenizer)
                
                # Create a new sentence by appending the predicted word
                new_seq = seq + ' ' + word
                
                # Calculate the score for the new sentence
                new_score = score - np.log(yhat[0][j])
                
                # Add the new sentence and its score to the candidates
                candidates.append([new_seq, new_score])
                
        # Sort the candidates by their scores
        ordered = sorted(candidates, key=lambda x: x[1])
    
        # Select the top k candidates as the new beam
        start = ordered[:beam_index]
    start.reverse()
   
    return start


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
        # Generate 5 captions using beam search
        caption = generate_desc(model, tokenizer, photo, max_length)
        final_sentences = []
        final_conf = []
        for i in range(0,len(caption)):
            final_sentences.append(caption[i][0][6:-4])
            final_conf.append(caption[i][1])
        print(final_sentences)
        print(final_conf)
        max = np.max(final_conf)
        max = max/5
        print(max+1)
        temperartur = (max+1)
        final_conf = final_conf/temperartur
        softmax_probabilities = np.exp(final_conf) / np.sum(np.exp(final_conf))
    
        softmax_probabilities = np.round(softmax_probabilities, 2)
        print(softmax_probabilities)
        blank_image = img.copy()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imshow("Image",img)
        cv2.waitKey(0)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()

