# 🛡️ VisioGuard

An AI-powered real-time anomaly detection and surveillance system that monitors live CCTV and webcam feeds to detect weapons and violent activities. VisioGuard leverages deep learning models to identify threats, log incidents, capture evidence, and send instant notifications, helping improve security through intelligent video surveillance.

---

## 📖 Overview

VisioGuard is an intelligent surveillance system developed as a Final Year B.E. Computer Engineering major project. It combines computer vision, deep learning, and web technologies to detect weapons (guns and knives) and violent activities in real time.

The system continuously monitors live video streams and, upon detecting an anomaly, automatically:

- 🔫 Detects weapons using a custom-trained YOLOv11 model
- 👊 Detects fight-like activities using a trained fight detection model
- 📸 Captures evidence images
- 📝 Logs events into an SQLite database
- 🚨 Sends instant Pushbullet notifications
- 🌐 Displays detections on a Flask-based dashboard

VisioGuard is designed to be lightweight, efficient, and scalable, making it suitable for real-time surveillance applications.

---

## ✨ Features

- 🔫 Real-time Gun and Knife Detection
- 👊 Real-time Fight Detection
- 📹 Live Webcam/CCTV Monitoring
- 🌐 Interactive Flask Dashboard
- 🚨 Instant Pushbullet Notifications
- 📝 SQLite Event Logging
- 📸 Automatic Evidence Image Capture
- ⚡ Optimized Real-Time Video Processing
- 🎯 Configurable Detection Parameters

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Backend | Flask |
| Computer Vision | OpenCV |
| Object Detection | YOLOv11 |
| Fight Detection | Deep Learning Model |
| Database | SQLite |
| Notifications | Pushbullet |
| Frontend | HTML, CSS, JavaScript |

---

## 📂 Project Structure

```text
VisioGuard/
│
├── app.py
├── notification.py
├── yolotest.py
├── fightTest.py
├── requirements.txt
├── README.md
├── visioguard.db
├── static/
├── templates/
└── logs/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/janki232004/VisioGuard.git
cd VisioGuard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the trained model files

Before running the application, place the following trained model files in the project directory:

- `best.pt`
- `model_16_m3_0.8888.pth`

### 4. Run the application

```bash
python app.py
```

### 5. Open the application

Visit:

```
http://127.0.0.1:5000
```

---

## 📦 Model Files

The trained model weights:

- `best.pt`
- `model_16_m3_0.8888.pth`

are **not included** in this repository because GitHub's web interface limits uploads to 25 MB per file.

Please place your trained model files in the project directory before running the application.

---

## 🔧 Environment Variables

Configure the following environment variables if required.

```powershell
$env:CAMERA_SOURCE="0"
$env:CAMERA_BACKEND="dshow"
$env:WEAPON_MODEL="best.pt"
$env:ALERT_COOLDOWN_SECONDS="30"
$env:DETECT_EVERY_N_FRAMES="5"
$env:CONFIDENCE_THRESHOLD="0.35"
$env:YOLO_IMAGE_SIZE="416"
$env:STREAM_DELAY_SECONDS="0.08"
$env:WEAPON_DETECTION_ENABLED="1"
$env:FIGHT_DETECTION_ENABLED="1"
$env:FIGHT_EVERY_N_FRAMES="10"
$env:FIGHT_MOTION_THRESHOLD="2.2"
$env:FIGHT_AREA_THRESHOLD="0.08"
$env:FIGHT_CONFIRMATION_WINDOWS="3"
$env:USE_FIGHT_LIBRARY="1"
$env:VISIOGUARD_SECRET_KEY="replace-with-a-random-secret"
$env:PUSHBULLET_TOKEN="your-pushbullet-token"
```

`CAMERA_SOURCE` can be either:

- Webcam index (`0`, `1`, etc.)
- IP Camera URL

---

## ⚡ Performance Optimization

To improve real-time performance on lower-end hardware:

- Increase `DETECT_EVERY_N_FRAMES`
- Increase `FIGHT_EVERY_N_FRAMES`
- Reduce `YOLO_IMAGE_SIZE` to `320`
- Increase `YOLO_IMAGE_SIZE` to `640` for higher accuracy
- Increase `ALERT_COOLDOWN_SECONDS` to reduce repeated notifications

---

## 🔔 Notifications

VisioGuard integrates **Pushbullet** for real-time alert delivery.

When an anomaly is detected, the system:

- Captures an evidence image
- Records the event in the SQLite database
- Sends an instant Pushbullet notification
- Updates the monitoring dashboard

Before running the application, configure your Pushbullet Access Token using the `PUSHBULLET_TOKEN` environment variable.

---

## 🚀 Future Enhancements

- Multi-camera surveillance support
- Cloud database integration
- Mobile application
- Face recognition
- Person tracking
- Email and SMS notifications
- AI-powered analytics dashboard
- User authentication and role-based access

---

## 👨‍💻 Authors

**VisioGuard** was developed as a Final Year B.E. Computer Engineering Major Project.

**Team Members**

- Janki Palpattuwar
- Arya Mokashi
- Nisha Lohar
- Sakshi Pawar

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

Feel free to fork this repository, open issues, or submit pull requests to help improve the project.

---

## 📄 License

This project is developed for educational and research purposes only.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!