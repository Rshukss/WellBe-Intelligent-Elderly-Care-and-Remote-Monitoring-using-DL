# WellBe: Intelligent Elderly Care & Remote Monitoring

WellBe is a proposed AI system (proof-of-concept) for monitoring elderly individuals, combining deep learning, pose estimation, and a smart health chatbot. It provides real-time fall detection, health query assistance, and a user-friendly web dashboard for caregivers and families.

---

## Features

- **Real-Time Fall Detection:** Uses YOLOv7 pose estimation to detect falls and alert caregivers.
- **Health Chatbot:** AI-powered assistant for health-related queries, BMI calculation, and general support.
- **Web Dashboard:** Live video feed, chatbot access, and user management in a single interface.
- **Automated Alerts:** Sends email notifications with images when a fall is detected.
- **Data Logging:** Tracks events and user interactions for review and analysis.

---

## Prerequisites

- **Python 3.x** (recommended)
- **pip** (Python package installer)

---

## Installation

1. **Clone the Repository**
   ```sh
   git clone https://github.com/Rshukss/WellBe-Intelligent-Elderly-Care-and-Remote-Monitoring-using-DL.git
   cd WellBe-Intelligent-Elderly-Care-and-Remote-Monitoring-using-DL
   ```

2. **Install Required Python Packages**
   ```sh
   pip install -r requirements.txt
   ```

3. **Download the Pose Estimation Model**
   - Download `yolov7-w6-pose.pt` from [YOLOv7 Releases](https://github.com/WongKinYiu/yolov7/releases).
   - Place the file in the project root directory.

4. **(Optional) Download NLTK Data for Chatbot**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

---

## Running the Application

Start the Flask server:

```sh
python app.py
```

- The application will run in debug mode.
- Access the dashboard at: [http://localhost:5000/](http://localhost:5000/)

---

## Project Structure

```
WellBe/
│
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── yolov7-w6-pose.pt       # Pose estimation model (download separately)
├── static/                 # Static files (CSS, JS, images)
├── templates/              # HTML templates for Flask
├── ...                     # Other supporting scripts and models
```

---

## Usage

- **Login/Register:** Access the dashboard securely.
- **Live Monitoring:** View real-time video and fall detection alerts.
- **Chatbot:** Click the chatbot button for health advice and BMI calculation.
- **User Details:** Enter user info for personalized responses.

---

## Dependencies

All dependencies are listed in `requirements.txt`. Install them with:

```sh
pip install -r requirements.txt
```

---

## Acknowledgements

- [YOLOv7](https://github.com/WongKinYiu/yolov7) for pose estimation.
- [Flask](https://flask.palletsprojects.com/) for the web framework.
- [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) for chatbot AI.

---

## License

This project is for academic and research purposes only.

---

**For issues or contributions, please open an issue or submit a pull request.**
