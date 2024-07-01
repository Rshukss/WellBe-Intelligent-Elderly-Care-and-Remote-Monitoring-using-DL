from flask import Flask, render_template, Response, request, url_for, redirect,flash, session
import cv2
import torch
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, fall_acc
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from deepface import DeepFace
from torchvision import transforms
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage  
import smtplib
import math
import datetime 
import time 
import random
from textblob import TextBlob
from chatbot import get_bmi_category,correct_spelling,calculate_bmi,predict_class,getResponse, predict_class, getResponse
import pickle
import json
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from difflib import get_close_matches

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app.secret_key = 'daisf_sjsbjcsvj_42dsfsf_4353tdggd-dsdad-ak'
user_context = {}


# Email configuration for Gmail
EMAIL_FROM = 'afgdgdfsg@gmail.com'
EMAIL_TO = 'sshiroodkar@gmail.com'
EMAIL_PASSWORD = 'vmfp ufkh gcaa okre'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

valid_user = {'username': 'admin', 'email': 'user@me.com', 'password': '1234'}

# Configuration for image saving path
IMAGE_FOLDER_PATH = 'Fall images'
emails_sent_count = 0

def send_email(subject, body, image_path=None):
    global emails_sent_count

    try:
        # Check if the maximum allowed emails have been sent
        if emails_sent_count >= 5:
            print("Maximum emails sent. Waiting for 5 minutes before sending the next batch.")
            time.sleep(300)  # Wait for 5 minutes (300 seconds)
            emails_sent_count = 0  # Reset the count after the wait

        # Your existing code for sending emails
        message = MIMEMultipart()
        message['From'] = EMAIL_FROM
        message['To'] = EMAIL_TO
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        if image_path:
            with open(image_path, 'rb') as image_file:
                attachment = MIMEImage(image_file.read(), name='fall_detection.jpg')
            message.attach(attachment)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, message.as_string())

        print("Email sent successfully!")

        # Increment the count of sent emails
        emails_sent_count += 1

    except Exception as e:
        print(f"Error sending email: {e}")



def fall_detection(poses):
    for pose in poses:
        xmin, ymin = (pose[2] - pose[4] / 2), (pose[3] - pose[5] / 2)
        xmax, ymax = (pose[2] + pose[4] / 2), (pose[3] + pose[5] / 2)
        left_shoulder_y = pose[23]
        left_shoulder_x = pose[22]
        right_shoulder_y = pose[26]
        left_body_y = pose[41]
        left_body_x = pose[40]
        right_body_y = pose[44]
        len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
        left_foot_y = pose[53]
        right_foot_y = pose[56]
        dx = int(xmax) - int(xmin)
        dy = int(ymax) - int(ymin)
        difference = dy - dx
        if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
                len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or (
                right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (
                len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)) \
                or difference < 0:
            return True, (xmin, ymin, xmax, ymax)
    return False, None


def falling_alarm(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255),
                  thickness=5, lineType=cv2.LINE_AA)
    acc=fall_acc()
    res=f"Fall Detected {acc}"
    cv2.putText(image, res, (11, 100), 0, 1, [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)
accuracy=fall_acc()

def get_pose_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    _ = model.float().eval()
    if torch.cuda.is_available():
        model = model.half().to(device)
    return model, device

def get_pose(image, model, device):
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    if torch.cuda.is_available():
        image = image.half().to(device)
    with torch.no_grad():
        output, _ = model(image)
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'],
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    return image, output


def prepare_image(image):
    _image = image[0].permute(1, 2, 0) * 255
    _image = _image.cpu().numpy().astype(np.uint8)
    _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
    return _image

    
def process_webcam():
    cap = cv2.VideoCapture(1)
    model, device = get_pose_model()
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    frame_count = 0
    emotion_interval = 10
    fall_start_time = None
    _image = None  

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to capture frame")
            break

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                color = (0, 255, 0)
                thickness = 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

                if frame_count % emotion_interval == 0:
                    emotion_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    emotion = emotion_result[0]['dominant_emotion']
                    cv2.putText(frame, emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)
                frame_count += 1
        except ValueError as e:
            print(f"Warning: {e}")

        if frame_count >= 20 * 5:
            image, output = get_pose(frame, model, device)
            _image = prepare_image(image)
            is_fall, bbox = fall_detection(output)

            if is_fall:
                if fall_start_time is None:
                    fall_start_time = time.time()
                
                fall_duration = time.time() - fall_start_time

                if fall_duration > 5:
                    falling_alarm(_image, bbox)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = f"{IMAGE_FOLDER_PATH}/fall_detection_{timestamp}.jpg"
                    cv2.imwrite(image_path, _image)
                    send_email("Fall Detected", "A fall has been detected. Check the camera feed.", image_path)
            else:
                fall_start_time = None

        else:
            _image = frame.copy()

        if _image is not None:
            _, jpeg = cv2.imencode('.jpg', _image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        entered_username = request.form['username']
        entered_email = request.form['email']
        entered_password = request.form['password']

        if (
            entered_username == valid_user['username'] and
            entered_email == valid_user['email'] and
            entered_password == valid_user['password']
        ):
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Login unsuccessful. Please check your credentials.', 'danger')

    return redirect(url_for('login_page'))


@app.route('/monitoring')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/chatbot")
def home():
    # Check if user information is available in the session
    if "user_info" in session:
        return render_template("chat.html", user_info=session["user_info"])
    else:
        return redirect(url_for("user_details"))


@app.route("/get", methods=["POST"])
def chatbot_response():
    try:
        msg = request.form["msg"]

        # Correct spelling mistakes in the message
        msg_words = nltk.word_tokenize(msg)
        corrected_msg = " ".join(correct_spelling(word) for word in msg_words)

        # Check if there is an ongoing conversation
        if "context" in user_context:
            corrected_msg = user_context["context"] + " " + corrected_msg

        # Check if user information is available in the session
        if "user_info" in session:
            user_info = session["user_info"]
            # You can now use user_info['name'], user_info['age'], etc. in your responses.

        if corrected_msg.lower() == 'start':
            # Clear user context when the conversation starts
            user_context.clear()

            # Ask for user information
            return "Hello! I'm your personal assistant. To get started, please tell me your name."

        elif "bmi" in corrected_msg.lower():
            if "height" in user_info and "weight" in user_info:
                bmi = calculate_bmi(float(user_info["height"]), float(user_info["weight"]))
                bmi_category = get_bmi_category(bmi)
                return f"Your BMI is: {bmi:.2f}\nBMI Category: {bmi_category}"
            else:
                return "I need your height and weight to calculate your BMI. Please provide that information first."
        
        else:
            ints = predict_class(corrected_msg, model)
            res = getResponse(ints, intents)

        return res
    except Exception as e:
        # Handle unexpected errors
        print(e)
        return "Sorry, something went wrong. Please try again later."


@app.route("/user_details", methods=["GET"])
def user_details():
    return render_template("user_details.html")



@app.route("/save_user_details", methods=["POST"])
def save_user_details():
    name = request.form.get("name")
    age = request.form.get("age")
    height = request.form.get("height")
    weight = request.form.get("weight")

    # Save user details to the session or database
    user_info = {"name": name, "age": age, "height": height, "weight": weight}
    session["user_info"] = user_info

    # Redirect to the home page or wherever you want
    return redirect(url_for("home"))




if __name__ == '__main__':
    app.run(debug=True)