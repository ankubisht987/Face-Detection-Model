import cv2
import numpy as np

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Extract face region and predict identity
        face_region = gray_image[y:y+h, x:x+w]  
        id, confidence = clf.predict(face_region)  
        confidence = int(100 * (1 - confidence / 300))

        # Assign labels based on ID
        if confidence > 77:
            label = "UNKNOWN"
            if id == 1:
                label = "Ankush"
            elif id == 2:
                label = "Maa"
            # elif id == 3:
            #     label = "Aniket"

            cv2.putText(img, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "UNKNOWN", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        coords = [x, y, w, h]
    return coords

def recognize(img, clf, faceCascade):
    draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), clf)
    return img

# Load face detection and recognition models
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, img = video_capture.read()
    if not ret:
        print("Failed to capture image")
        break

    img = recognize(img, clf, faceCascade)
    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) & 0xFF == 13:  # Press Enter key to exit
        break

video_capture.release()
cv2.destroyAllWindows()
