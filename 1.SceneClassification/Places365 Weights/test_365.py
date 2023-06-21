import os
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np

model_path = 'tf_googleNetPlaces365'
glob_correct = 0
glob_all = 0

names = []
# Open the file for reading
with open("label_names.txt", "r") as f:
    # Loop through each line in the file
    for line in f:
        # Split the line into name and index
        name, index = line.strip().split()
        # Convert the index to an integer
        index = int(index)
        # If the index is greater than or equal to the length of the list,
        # append None values to the list until the list is long enough
        while index >= len(names):
            names.append(None)
        # Set the value at the specified index to the name
        names[index] = name

# Print the resulting list of names
print(names)

def predict_category(img_name):
    with tf.Session() as sess:
        tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)

        # Get the input and output tensors
        input_tensor = sess.graph.get_tensor_by_name('data:0')
        output_tensor = sess.graph.get_tensor_by_name('prob:0')

        # Load the input image
        image_path = 'val/' + img_name
        #print(image_path)
        image = cv2.imread(image_path)

        # Resize the image to match the input shape of the model
        input_shape = input_tensor.shape.as_list()[1:3]
        image_resized = cv2.resize(image, tuple(input_shape[::-1]))

        # Expand the image dimensions to create a batch of one image
        image_expanded = image_resized[np.newaxis, ...]

        # Run the prediction
        prediction = sess.run(output_tensor, feed_dict={input_tensor: image_expanded})

        # Get the top 5 predicted class indices and probabilities
        top_k = np.argsort(prediction)[0, ::-1][:5]
        class_indices = top_k
        probabilities = prediction[0, class_indices]

        # Get the class labels for the top 5 predicted class indices
        labels_ = []
        
        for class_index in class_indices:
            #print(names[class_index])
            labels_.append(names[class_index])

        # Return the top 5 class labels and probabilities
        return labels_, probabilities

labels = []

# Set the number of photos to test
num_test_photos = 3

# Set the path to your folders
folder_path = "val/"

# Create a list of all the folder names
folder_names = os.listdir(folder_path)

# Create a dictionary to store the number of correct and total predictions for each category
results = {name: {'correct': 0, 'total': 0} for name in folder_names}

# Create an empty list to hold the names
names = []

# Open the file for reading
with open("label_names.txt", "r") as f:
    # Loop through each line in the file
    for line in f:
        # Split the line into name and index
        name, index = line.strip().split()
        # Convert the index to an integer
        index = int(index)
        # If the index is greater than or equal to the length of the list,
        # append None values to the list until the list is long enough
        while index >= len(names):
            names.append(None)
        # Set the value at the specified index to the name
        names[index] = name

# Print the resulting list of names
print(names)

i = 1
# Loop through each folder
for folder_name in folder_names:
    folder = os.path.join(folder_path, folder_name)
    
    # Get a list of all the image filenames in the folder
    try:
        image_filenames = os.listdir(folder)
    except:
        continue
        
    # Shuffle the list of image filenames
    random.shuffle(image_filenames)
    
    # Get a random sample of the image filenames
    test_image_filenames = image_filenames[:num_test_photos]
    
    # Loop through each test image
    for test_image_filename in test_image_filenames:
        # Determine the actual category of the image based on the folder name
        actual_category = folder_name

        img_name = os.path.join(folder_name,test_image_filename)        
        predicted_category, scores = predict_category(img_name)
        print(predicted_category)
        print(scores)
        break
        predicted_category = names[predicted_category_index]
        glob_all+=1
        # Increment the correct prediction count if the predicted category matches the actual category
        if predicted_category == actual_category:
            results[folder_name]['correct'] += 1
            glob_correct +=1
            print(glob_correct,"/",glob_all)
            predict_color = (0,255,0)
        
        else:
            print("Correct is : {}. predicted was : {}.".format(actual_category,predicted_category))
            predict_color = (0,0,255)
        
        # Increment the total prediction count
        results[folder_name]['total'] += 1
        img = cv2.imread('val/' + img_name)
        img = cv2.resize(img,(500,500))
        img = cv2.putText(img, "Correct Label:{}".format(actual_category), (10,50), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 2)
        img = cv2.putText(img, "Predicted Label:{}".format(predicted_category), (10,100), cv2.FONT_HERSHEY_SIMPLEX, .5, predict_color, 2)
        if predicted_category != actual_category:
            print("writing:")
            cv2.imwrite("{}.jpeg".format(i),img)
            i+=1
        cv2.imshow("Image",img)
        cv2.waitKey(0)
        

# Calculate the overall test accuracy
total_correct = sum([results[name]['correct'] for name in folder_names])
total_tested = sum([results[name]['total'] for name in folder_names])
overall_accuracy = total_correct / total_tested

# Print the results for each category
for name in folder_names:
    print("folder_names: ",folder_names)
    print("name: ",name)
    if results[name]['total'] > 0:
        correct = results[name]['correct']
        total = results[name]['total']
        accuracy = correct / total
        print(f"Category: {name} - Accuracy: {accuracy:.2f} ({correct}/{total} correct predictions)")
    else:
        print(f"Category: {name} - No images found")

# Print the overall test accuracy
print(f"Overall test accuracy: {overall_accuracy:.2f} ({total_correct}/{total_tested} correct predictions)")
