from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
import os
import uuid

# Define the folder for storing unknown face images
UNKNOWN_FACES_FOLDER = 'unknown_faces'
if not os.path.exists(UNKNOWN_FACES_FOLDER):
    os.makedirs(UNKNOWN_FACES_FOLDER)

# Initialize MTCNN and Inception Resnet V1
mtcnn = MTCNN(keep_all=True)
inception = InceptionResnetV1(pretrained='vggface2').eval()

def get_face_embeddings(image):
    boxes, probs = mtcnn.detect(image)
    if boxes is not None:
        faces = mtcnn(image)
        embeddings = inception(faces).detach().numpy()
        return embeddings
    else:
        return []

# Load known faces
known_face_encodings = []
known_face_names = []

def load_known_faces(known_faces_folder):
    for filename in os.listdir(known_faces_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_folder, filename)
            image = cv2.imread(image_path)
            embeddings = get_face_embeddings(image)
            if len(embeddings) > 0:
                known_face_encodings.append(embeddings[0])
                known_face_names.append(os.path.splitext(filename)[0])

# Load known faces from the folder
load_known_faces('known_faces')

# Initialize the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get embeddings
    boxes, _ = mtcnn.detect(rgb_frame)
    if boxes is not None:
        faces = mtcnn(rgb_frame)
        embeddings = inception(faces).detach().numpy()

        for i, (box, embedding) in enumerate(zip(boxes, embeddings)):
            x1, y1, x2, y2 = map(int, box)
            distances = np.linalg.norm(np.array(known_face_encodings) - embedding, axis=1)
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                name = "Unknown"

                if distances[best_match_index] < 0.6:  # Threshold can be adjusted
                    name = known_face_names[best_match_index]
                else:
                    # Save unknown face to a file
                    unique_filename = os.path.join(UNKNOWN_FACES_FOLDER, f"unknown_{uuid.uuid4().hex}.jpg")
                    cv2.imwrite(unique_filename, rgb_frame[y1:y2, x1:x2])

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw a label with the name
                cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (x1 + 6, y1 - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
video_capture.release()
cv2.destroyAllWindows()
