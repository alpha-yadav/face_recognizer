import cv2
import sqlite3 as sql
import os
person_name = input("Enter the name of the person: ")
os.makedirs('DB', exist_ok=True)
if not person_name:
    raise ValueError("Person name cannot be empty.")
sql_connection = sql.connect('DB/face.db')
cursor = sql_connection.cursor()
cursor.execute('''PRAGMA foreign_keys = ON''')
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
               )''')
cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
    _id INTEGER PRIMARY KEY AUTOINCREMENT,
    u_id INTEGER NOT NULL,
    image BLOB NOT NULL
);''')
cursor.execute("INSERT INTO users (name) VALUES (?)", (person_name,))
sql_connection.commit()
cursor.execute("SELECT id FROM users WHERE name = ?", (person_name,))
person_id = cursor.fetchone()[0]
cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img_id = 0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img_id += 1
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))
        img_d=cv2.imencode(".jpg", face)
        img_d = img_d[1].tobytes()
        cursor.execute("INSERT INTO faces (u_id, image) VALUES (?, ?)", (person_id, img_d))
        sql_connection.commit()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Capturing Faces", img)

    if cv2.waitKey(1) == 27 or img_id >= 250:  # ESC or 50 images
        break
sql_connection.close()
cap.release()
cv2.destroyAllWindows()
