from flask import Flask, request, render_template
from flask_cors import CORS
from email_sender import send_email
from image_analyzer import get_indications

app = Flask(__name__, static_folder="static", template_folder="templates")
cors = CORS(app, resources={r"*": {"origins": "*"}}, supports_credentials=True)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/api/send_email/", methods=["POST"])
def send_email_handler():
    data = request.get_json()
    subject = data["subject"]
    body = data["body"]
    target = data["target"]
    send_email(subject, body, target)
    return "Email sent"

@app.route("/api/analyze_photo/", methods=["POST"])
def analyze_photo_handler():
    image = request.form["file"]
    get_indications(image)
    return "Photo analyzed"