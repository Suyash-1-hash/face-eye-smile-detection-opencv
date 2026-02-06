# Face, Eye, and Smile Detection using OpenCV

This project is a real-time computer vision application built using **Python** and **OpenCV**. It detects **faces, eyes, and smiles** from a live webcam feed using **Haar Cascade classifiers**, which are pre-trained models provided by OpenCV.

The project demonstrates the practical use of image processing and object detection techniques and serves as a strong foundation for advanced applications such as face recognition and smart attendance systems.

---

## 🚀 Features

- Real-time face detection using webcam
- Eye detection within detected face region
- Smile detection within detected face region
- Bounding boxes drawn around detected faces
- Text labels displayed when eyes or smile are detected
- Uses lightweight, pre-trained Haar Cascade models

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV (cv2)**
- **Haar Cascade Classifiers**

---

## 📂 Project Structure

face-eye-smile-detection-opencv/<br>
│<br>
├── main.py<br>
├── haarcascades/<br>
│ ├── haarcascade_frontalface_default.xml<br>
│ ├── haarcascade_eye.xml<br>
│ └── haarcascade_smile.xml<br>
│<br>
├── requirements.txt<br>
└── README.md<br>