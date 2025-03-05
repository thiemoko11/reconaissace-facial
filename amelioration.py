import face_recognition
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator




dossier = 'known_faces'
if not os.path.exists(dossier):
    os.makedirs(dossier)


nom_personne = input("Entrez le nom de la personne : ")
dossier_personne = os.path.join(dossier, nom_personne)

if not os.path.exists(dossier_personne):
    os.makedirs(dossier_personne)


cap = cv2.VideoCapture(0)


count = 0

while count < 20:  
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la capture de l'image")
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

   
    for (x, y, w, h) in faces:
       
        face = frame[y:y + h, x:x + w]

       
        image_path = os.path.join(dossier_personne, f"{nom_personne}_{count}.jpg")
        cv2.imwrite(image_path, face)
        count += 1

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    
    cv2.imshow('Capturer des visages', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


dossier_augmente = 'augmented_faces'
if not os.path.exists(dossier_augmente):
    os.makedirs(dossier_augmente)


datagen = ImageDataGenerator(
    rotation_range=30,       
    width_shift_range=0.2,   
    height_shift_range=0.2,  
    shear_range=0.2,         
    zoom_range=0.2,         
    horizontal_flip=True,    
    fill_mode='nearest'      
)


for nom_personne in os.listdir('known_faces'):
    dossier_personne = os.path.join('known_faces', nom_personne)

    if not os.path.isdir(dossier_personne):
        continue 

    
    for image_name in os.listdir(dossier_personne):
        image_path = os.path.join(dossier_personne, image_name)

        if not image_name.endswith(".jpg"):
            continue  

        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (160, 160))  

       
        image = np.expand_dims(image, axis=0)

        
        i = 0
        for batch in datagen.flow(image, batch_size=1, save_to_dir=dossier_augmente, save_prefix=nom_personne, save_format='jpg'):
            i += 1
            if i > 20:  
                break


known_faces = []
known_names = []

for nom_personne in os.listdir('known_faces'):
    dossier_personne = os.path.join('known_faces', nom_personne)
    if not os.path.isdir(dossier_personne):
        continue

    for filename in os.listdir(dossier_personne):
        if filename.endswith('.jpg'):
            image_path = os.path.join(dossier_personne, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:  
                known_faces.append(encodings[0])
                known_names.append(nom_personne)

if not known_faces:
    print("Aucun visage connu chargé. Vérifiez les images dans 'known_faces'.")
    exit()


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Erreur lors de la capture de la vidéo")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        if face_distances.size == 0:  
            continue

        best_match_index = np.argmin(face_distances)
        name = "Inconnu"
        if face_distances[best_match_index] < 0.6: 
            name = known_names[best_match_index]

        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

   
    cv2.imshow('Reconnaissance Faciale', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
