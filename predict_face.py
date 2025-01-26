import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from utils import load_class_names

def predict_face(model, class_names):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            face_img = frame[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (150, 150))
            face_img_array = image.img_to_array(face_img_resized) / 255.0
            face_img_array = np.expand_dims(face_img_array, axis=0)
            
            prediction = model.predict(face_img_array)
            predicted_class_index = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_index]
            predicted_person = class_names[predicted_class_index]

            if confidence > 0.8:
                cv2.putText(frame, f"{predicted_person} ({confidence*100:.2f}%)", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Alien", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('n'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = load_model("path to ai where model .h5 file is saved")
    class_names = load_class_names("path for the dataset")
    predict_face(model, class_names)
