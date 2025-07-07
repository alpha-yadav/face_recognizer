import cv2
import numpy as np
import sqlite3 as sql
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
sql_connection = sql.connect('DB/face.db')
cursor = sql_connection.cursor()
cursor.execute("SELECT u_id, image FROM faces")
rows = cursor.fetchall()
if not rows:
    raise ValueError("No faces found in the database. Please collect faces first.")
namex=''
for u_id, img_data in rows:
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        faces.append(img)
        labels.append(u_id-1)
recognizer.train(faces, np.array(labels))
recognizer.save("trained_data.yml")
