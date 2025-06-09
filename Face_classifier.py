import numpy as np
from PIL import Image
import os
import cv2

# Function to train the face recognition classifier
def train_classifier(data_dir):
    if not os.path.exists(data_dir):
        print(f"Error: The directory '{data_dir}' does not exist.")
        return

    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        try:
            img = Image.open(image).convert('L')  # Convert to grayscale
            imageNp = np.array(img, 'uint8')  # Convert image to numpy array
            id = int(os.path.split(image)[1].split(".")[1])  # Extract ID from filename

            faces.append(imageNp)
            ids.append(id)
        except Exception as e:
            print(f"Skipping file {image}: {e}")

    if len(faces) == 0:
        print("Error: No valid face images found in the dataset!")
        return

    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    print("✅ Training complete. Classifier saved as 'classifier.xml'.")

# Run the function
train_classifier("data")  # Ensure this folder contains face images
