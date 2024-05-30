# Import necessary libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize the images to values between 0 and 1
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Class names corresponding to CIFAR-10 labels
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Display the first 16 images from the training dataset
for i in range(16):
    plt.subplot(4, 4, i + 1)  # Create a 4x4 grid of subplots
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.imshow(training_images[i], cmap=plt.cm.binary)  # Display image
    plt.xlabel(class_names[training_labels[i][0]])  # Label the image with its class name

plt.show()

# Reduce the dataset size for quicker training
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Uncomment the following block to define, compile, and train the model
"""
# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # First convolutional layer
model.add(layers.MaxPooling2D(2, 2))  # First max pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Second convolutional layer
model.add(layers.MaxPooling2D(2, 2))  # Second max pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Third convolutional layer
model.add(layers.Flatten())  # Flatten the output
model.add(layers.Dense(64, activation='relu'))  # Fully connected layer
model.add(layers.Dense(10, activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# Evaluate the model
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save the trained model
model.save('image_classifier.keras')
"""

# Load the pre-trained model
model = models.load_model('image_classifier.keras')

# Load and preprocess the input image for prediction
img = cv.imread('rsz_horse.jpg')  # Read the image
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert image from BGR to RGB format

# Display the input image
plt.imshow(img, cmap=plt.cm.binary)

# Resize the image to 32x32 (the input size for the model)
img_resized = cv.resize(img, (32, 32))

# Make a prediction
prediction = model.predict(np.array([img_resized]) / 255.0)  # Normalize and reshape the image
index = np.argmax(prediction)  # Get the index of the highest predicted probability
print(f'Prediction is {class_names[index]}')  # Print the predicted class name
