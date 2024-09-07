# in this verison 2. i made it such that the camera window opens, and it recognizes one face if there are multiple faces (in the future
# version ill try implementing multiple faces at once.) for now it recognizes one face, once the user clicks "s" the face gets captured
# and the photo is captured. then the user has to enter his name in the terminal for now. ill try to implement a front-end so that it 
# all looks good and presentable. once the name is entered, the captured face is saved in the face_db folder


import cv2
import os
from deepface import DeepFace

# Load pre-trained face detector from OpenCV (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# switching on the webcam 
cap = cv2.VideoCapture(0)

# checking for webcame turns on normally or not
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# file where im saving the faces
db_path = 'face_db'
if not os.path.exists(db_path):
    os.makedirs(db_path)

# creating the window to display the face
cv2.namedWindow("Face Capture", cv2.WINDOW_AUTOSIZE)

def capture_and_save_face():
    face_detected = False
    message_printed = False  # Ensure the message is only printed once

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Resize frame to make processing faster 
        frame_small = cv2.resize(frame, (640, 480))

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame and draw a box around the face indicating that the users face is detected
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0 and not face_detected:
            # Focus on the first detected face
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame_small, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Only print the message once when the face is first detected
            if not message_printed:
                print("Face detected. Press 's' to save the face.")
                message_printed = True  # Prevent further prints
            
            # Display the camera feed with the rectangle
            cv2.imshow('Face Capture', frame_small)

            # Capture keypress from user to save the face
            key = cv2.waitKey(1) & 0xFF
            
            # Check if 's' is pressed to save the face
            if key == ord('s'):
                face_img = frame[y:y+h, x:x+w]  # Crop the face from the frame
                
                # Ask for user name and save the face
                name = input("Enter your name: ")
                face_path = f"{db_path}/{name}.jpg"
                cv2.imwrite(face_path, face_img)
                print(f"Saved {name}'s face.")
                
                # Encode and store the face using DeepFace
                DeepFace.represent(img_path=face_path, model_name="VGG-Face", enforce_detection=False)
                print(f"Encoded {name}'s face.")
                
                face_detected = True  # Stop detecting new faces after capturing one
            
            # Check if 'q' is pressed to quit
            if key == ord('q'):
                break
        
        # Display the camera feed
        cv2.imshow('Face Capture', frame_small)

    cap.release()
    cv2.destroyAllWindows()

# Run the face capture function
capture_and_save_face()

