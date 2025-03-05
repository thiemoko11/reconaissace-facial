from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app)  # Permet au frontend d'accéder à l'API

# Charger les visages connus
known_faces = []
known_names = []

def charger_visages():
    """Fonction pour charger les visages connus"""
    global known_faces, known_names
    known_faces = []
    known_names = []

    dossier_faces = "known_faces"
    if not os.path.exists(dossier_faces):
        os.makedirs(dossier_faces)  # Créer le dossier s'il n'existe pas
        print(" Dossier 'known_faces' créé. Ajoutez des images pour la reconnaissance.")
        return

    print("🔍 Chargement des visages connus...")
    for nom_personne in os.listdir(dossier_faces):
        dossier_personne = os.path.join(dossier_faces, nom_personne)
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
                else:
                    print(f" Aucun visage détecté dans {image_path}, image ignorée.")

    if not known_faces:
        print("Aucun visage connu chargé. Ajoutez des images valides dans 'known_faces'.")

# Charger les visages au démarrage
charger_visages()

@app.route('/reload', methods=['GET'])
def reload_faces():
    """Recharger les visages connus sans redémarrer l'API"""
    charger_visages()
    return jsonify({"message": "Visages rechargés avec succès"}), 200

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        if len(face_distances) == 0:
            continue

        best_match_index = np.argmin(face_distances)
        name = "Inconnu"
        if face_distances[best_match_index] < 0.6:
            name = known_names[best_match_index]

        results.append({"name": name, "location": [top, right, bottom, left]})

    return jsonify({"faces": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
