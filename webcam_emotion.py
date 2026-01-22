import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model("emotion_recognition_model.keras")


emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


cap = cv2.VideoCapture(0)


emotion_buffer = []

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        
        if face.size == 0:
            continue

        
        face = cv2.resize(face, (48, 48))
        face = np.reshape(face, (1, 48, 48, 1))


        
        prediction = model.predict(face, verbose=0)

        
        emotion_buffer.append(np.argmax(prediction))
        if len(emotion_buffer) > 5:
            emotion_buffer.pop(0)

        emotion_index = max(set(emotion_buffer), key=emotion_buffer.count)
        emotion = emotion_labels[emotion_index]

        confidence = np.max(prediction) * 100
        label = f"{emotion} ({confidence:.1f}%)"

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    
    cv2.imshow("Emotion Recognition", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
