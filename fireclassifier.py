import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
# Load the pre-trained model
model = load_model('fire_detection_model.h5')

# Function to preprocess the image
# def preprocess_image(image_path):
#     # Load the image
#     # Resize the image to match the input size of the model
#     # Convert the image to a float array and normalize pixel values
#     # Expand the dimensions of the image to match the input shape of the model
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (224, 224))
#     image = image.astype('float32') / 255.0
#     image = np.expand_dims(image, axis=0)
#     return image

# Function to classify the image
def classify_image(image):
    # image = preprocess_image(image)
    # Get the prediction probabilities
    probabilities = model.predict(image)[0]
    # Get the predicted class label
    predicted_class = np.argmax(probabilities)

    labels = ['fire', 'no fire']
    # Get the predicted class name
    predicted_class_name = labels[predicted_class]
    confidence = probabilities[predicted_class]
    return predicted_class_name, confidence

# Accept image input from the user
# image = cv2.imread('fire.1.png')  
image = cv2.imread('non_fire.1.png')  
cv2.imshow('Image',image)
# image = Image.open('fire.1.png')
image = cv2.resize(image, (224, 224))
image = image.astype('float32') / 255.0
image = np.expand_dims(image, axis=0)
# image_path = input("Enter the path of the image to be classified: ")
predicted_class_name, confidence = classify_image(image)
print("Predicted Class: ", predicted_class_name)
print("Confidence: ", confidence)