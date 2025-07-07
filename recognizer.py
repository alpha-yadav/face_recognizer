import cv2
import sqlite3 as sql
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_data.yml")
sql_connection = sql.connect('DB/face.db')
cursor = sql_connection.cursor()
cursor.execute("SELECT name FROM users")
names = cursor.fetchall()
names = [name[0] for name in names if name[0]]  # Ensure
if not names:
    raise ValueError("No names found in the database. Please collect faces first.")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
        label, confidence = recognizer.predict(roi)

        if confidence < 60:
            name = names[label]
            color = (0, 125, 0)
        else:
            name = "X"
            color = (0, 0, 125)

        cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
