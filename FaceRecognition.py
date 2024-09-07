import cv2
import os
from deepface import DeepFace

# Load pre-trained face detector from OpenCV (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Directory to store face images and embeddings
db_path = 'face_db'
if not os.path.exists(db_path):
    os.makedirs(db_path)

# Create a named window
cv2.namedWindow("Face Capture", cv2.WINDOW_AUTOSIZE)

def capture_and_save_face():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Resize frame to make processing faster
        frame_small = cv2.resize(frame, (640, 480))

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Reduced scale factor and minNeighbors for quicker detection
        
        if len(faces) > 0:
            # If at least one face is detected, process the first face
            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w]  # Crop the face from the frame
            cv2.rectangle(frame_small, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Display the camera feed with rectangle
            cv2.imshow('Face Capture', frame_small)
            
            # Ask for user name and save the face
            name = input("Enter your name: ")
            face_path = f"{db_path}/{name}.jpg"
            cv2.imwrite(face_path, face_img)
            print(f"Saved {name}'s face.")
            
            # Encode and store the face using DeepFace
            DeepFace.represent(img_path=face_path, model_name="VGG-Face", enforce_detection=False)
            print(f"Encoded {name}'s face.")
            
            break  # Terminate after saving one face
        
        # Display the camera feed
        cv2.imshow('Face Capture', frame_small)
        
        # Add a small delay to prevent the program from freezing (reduce CPU load)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Using 30ms delay
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the face capture function
capture_and_save_face()
