import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


model = load_model('models/model40.h5')
labels = ['apartment', 'autorickshaw stand', 'bus_station-indoor', 'car_interior', 'eletronics_store', 'farm', 'forest_path', 'formal_garden', 'highway', 'hospital', 'hospital_room', 'hotel_room', 'house', 'lake', 'living_room', 'market-indoor', 'office', 'open_field', 'pharmacy', 'religious_procession', 'restaurant', 'river', 'rural_area', 'shopping_mall-indoor', 'street', 'supermarket', 'taxi', 'tea_stall', 'temple-asia', 'train_station-platform']


def predict_category(img_path,labels=labels):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)

    # Get the indices of the top 5 predicted classes
    top_5_indices = np.argsort(predictions[0])[::-1][:5]
    top_5_probabilities = predictions[0][top_5_indices]
    top_5_probabilities = np.round(top_5_probabilities, 3)
    # Get the labels corresponding to the top 5 predicted classes
    top_5_labels = [labels[i] for i in top_5_indices]

    return top_5_labels, top_5_probabilities



