# 🪖 AI-Integrated Smart Combat Helmet for Soldier Safety and Tactical Awareness

An Edge AI-based smart combat helmet designed to enhance soldier safety in harsh and network-denied environments using Computer Vision, Raspberry Pi 4, and IoT sensors. The system provides real-time threat detection, health monitoring, GPS tracking, and emergency voice alerts without relying on cloud connectivity.

---

## 📌 Project Overview

This project integrates Artificial Intelligence, Computer Vision, and IoT technologies to improve battlefield situational awareness and soldier safety. The system performs real-time object detection using YOLOv8, monitors vital health parameters, tracks GPS location, and provides emergency alerts while operating efficiently on edge devices.

---

## ✨ Key Features

- 🎯 Real-time object detection using YOLOv8
- ❤️ Heart rate and SpO₂ monitoring
- 📍 GPS location tracking
- 🔊 Emergency voice alert system
- 🧭 Motion detection using IMU sensor
- 🪖 Edge AI deployment on Raspberry Pi 4
- 📡 Designed for harsh and network-denied environments

---

## 🛠️ Tech Stack

### Programming Language
- Python

### AI & Computer Vision
- YOLOv8
- OpenCV
- Ultralytics
- NumPy

### Hardware
- Raspberry Pi 4
- Raspberry Pi Camera
- Neo-6M GPS
- MAX30100 Pulse Oximeter
- MPU6050 Accelerometer & Gyroscope
- Speaker

### Tools
- VS Code
- Jupyter Notebook
- Git
- GitHub

---

## 🏗️ System Architecture
<img width="1072" height="722" alt="image" src="https://github.com/user-attachments/assets/a4ecfb0a-9e9f-40ca-9f10-7293bf61b118" />


Example:

```
Pi Camera → YOLOv8 → Threat Detection
                     ↓
      Raspberry Pi 4 (Edge AI)
      ↓       ↓          ↓
 GPS   Health Sensor   Voice Alert
      ↓
 Emergency Information
```

---

## 📂 Project Structure

```
AI-Integrated-Smart-Combat-Helmet/
│
├── src/
├── hardware/
├── models/
├── screenshots/
├── docs/
├── demo/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Prernabodke31/AI-Integrated-Smart-Combat-Helmet.git
```

Go to the project folder

```bash
cd AI-Integrated-Smart-Combat-Helmet
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python src/main.py
```

---

## 📊 Experimental Results

- Overall System Performance: **92.31%**
- Real-time threat detection
- GPS location tracking
- Soldier health monitoring
- Emergency voice alert generation
- Successful Edge AI deployment on Raspberry Pi 4

---

## 📸 Project Images

<img width="612" height="461" alt="image" src="https://github.com/user-attachments/assets/7010cc02-753c-4b27-955e-f54b8a993825" />
<img width="596" height="450" alt="image" src="https://github.com/user-attachments/assets/9d925559-8f54-4b02-b1fa-13c9eb2a02b6" />
<img width="582" height="456" alt="image" src="https://github.com/user-attachments/assets/8466a08f-3076-4bf6-9807-15efbbcde441" />
<img width="577" height="458" alt="image" src="https://github.com/user-attachments/assets/45362c5b-e4a9-4010-8f8f-f35cdc6d0ad3" />
<img width="587" height="456" alt="image" src="https://github.com/user-attachments/assets/1348c66e-5a65-4a82-8534-4bda067cfef4" />
<img width="572" height="452" alt="image" src="https://github.com/user-attachments/assets/fccd606b-df5c-4b4f-88ac-898fdc278fc0" />

---

## 📄 Research Publication

**AI-Integrated Smart Combat Helmet for Soldier Safety in Harsh and Network-Denied Environments**

Published in the **International Journal of Sciences and Innovation Engineering (IJSCI), Volume 3, Issue 5, 2026.**

---

## 🔮 Future Enhancements

- Thermal camera integration
- Drone-assisted surveillance
- LoRa communication
- AI-powered threat prediction
- Satellite communication support

---

## 👩‍💻 Author

**Prerna Bodke**

B.E. Artificial Intelligence & Data Science

MET Institute of Engineering, Nashik, India

LinkedIn: https://www.linkedin.com/in/prerna-bodke

GitHub: https://github.com/Prernabodke31

---

## ⭐ Support

If you find this project useful, please consider giving it a ⭐ on GitHub.
