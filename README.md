
# 🏋️‍♂️ Push-Up Counter with YOLOv8 Pose Detection

Ce projet utilise **YOLOv8 Pose Estimation** pour détecter les mouvements de l'utilisateur à partir d'une webcam en temps réel, compter le nombre de **pompes effectuées** et fournir un **feedback vocal** sur la forme.

## 📸 Fonctionnalités

* 📍 Détection en temps réel des points clés du corps à l’aide de `yolov8n-pose.pt`.
* 🧠 Lissage des positions pour plus de stabilité (réduction du jitter).
* 📐 Calcul des angles des bras et du corps pour détecter correctement les mouvements de pompe.
* 🔊 Feedback vocal avec `pyttsx3` pour corriger la posture et encourager l'utilisateur.
* 🧮 Compteur automatique de pompes bien exécutées.

## 🛠️ Technologies utilisées

* Python
* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* OpenCV
* NumPy
* pyttsx3 (synthèse vocale)
* threading & queue (gestion parallèle de la voix)
* cvzone (affichage enrichi)

## ▶️ Utilisation

1. **Installer les dépendances** :

   ```bash
   pip install ultralytics opencv-python numpy pyttsx3 cvzone
   ```

2. **Télécharger le modèle YOLOv8 Pose** :
   Le script utilise automatiquement `yolov8n-pose.pt`, mais assurez-vous que le fichier est bien dans le répertoire ou installez-le avec Ultralytics :

   ```bash
   yolo export model=yolov8n-pose.pt
   ```

3. **Lancer le script** :

   ```bash
   python pushup_counter.py
   ```

4. **Commencez à faire des pompes !** La voix vous guidera.

## 📊 Critères d'une bonne pompe

* **Bras pliés à plus de 90° en bas**
* **Bras tendus à plus de 160° en haut**
* **Corps aligné (angle hanche-épaules-cheville)**

## 📁 Structure du code

* `calculate_angle()` : calcule les angles entre 3 points.
* `smooth_keypoints()` : stabilise la détection des articulations.
* `filter_confidence()` : filtre les points avec faible confiance.
* `draw_angle()` : affiche visuellement les angles sur l’image.
* `speak()` et `worker_speak()` : synthèse vocale dans un thread séparé.
* `main loop` : détection vidéo + comptage de pompes.

---

