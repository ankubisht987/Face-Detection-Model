import cv2
import os

# Load the face detection classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to detect and crop faces
def face_cropped(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    print(f"Faces detected: {len(faces)}")  # Debugging print

    cropped_faces = []
    for (x, y, w, h) in faces:
        cropped_faces.append(image[y:y+h, x:x+w])

    return cropped_faces if cropped_faces else None

# Function to get the next available image ID
def get_next_img_id(save_dir, user_id):
    existing_files = [f for f in os.listdir(save_dir) if f.startswith(f"user.{user_id}.")]
    if not existing_files:
        return 1  # Start from 1 if no images exist
    existing_ids = [int(f.split(".")[2]) for f in existing_files]
    return max(existing_ids) + 1  # Continue from the last image ID

# Function to generate dataset
def generate_dataset(user_id):
    cap = cv2.VideoCapture(0)

    # Ensure the directory exists
    save_dir = "data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_id = get_next_img_id(save_dir, user_id)  # Get the next available image ID

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        faces = face_cropped(frame)
        if faces is not None:
            for face in faces:  # Save multiple detected faces
                face = cv2.resize(face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                file_name_path = os.path.join(save_dir, f"user.{user_id}.{img_id}.jpg")
                cv2.imwrite(file_name_path, face)

                cv2.putText(face, str(img_id), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cropped Face", face)

                img_id += 1  # Increment image ID for the next image

                if img_id > 200:  # Stop when 200 images are collected
                    break

        if cv2.waitKey(1) == 13 or img_id > 200:  # Press 'Enter' or collect 200 images
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Collecting samples for User {user_id} is completed.")

# Run the dataset collection for a specific user ID
user_id = input("Enter User ID: ")
generate_dataset(user_id)
