import os
from flask import Flask, request
from predict import predict_video

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def render_home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Deepfake Detector</title>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                text-align: center;
                padding: 50px;
            }

            .container {
                max-width: 500px;
                margin: auto;
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            }

            h1 {
                margin-bottom: 10px;
            }

            p {
                opacity: 0.8;
                margin-bottom: 20px;
            }

            input[type="file"] {
                margin: 20px 0;
                padding: 10px;
                border-radius: 8px;
                border: none;
                background: white;
                color: black;
                width: 100%;
            }

            button {
                background: #ff7a18;
                border: none;
                padding: 12px 25px;
                color: white;
                font-size: 16px;
                border-radius: 8px;
                cursor: pointer;
                transition: 0.3s;
            }

            button:hover {
                background: #ff4b2b;
            }

            .footer {
                margin-top: 20px;
                font-size: 12px;
                opacity: 0.6;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Deepfake Detector</h1>
            <p>Upload a video and detect if it's REAL or FAKE using AI</p>

            <form method="POST" action="/predict" enctype="multipart/form-data">
                <input type="file" name="file" required>
                <br>
                <button type="submit">🚀 Analyze</button>
            </form>

            <div class="footer">
                Mini Project
            </div>
        </div>
    </body>
    </html>
    """


def render_result(label, confidence):
    color = "#ff4b2b" if label == "FAKE" else "#00c853"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Result</title>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #1f4037, #99f2c8);
                color: white;
                text-align: center;
                padding: 50px;
            }}

            .card {{
                max-width: 500px;
                margin: auto;
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            }}

            .result {{
                font-size: 40px;
                margin: 20px 0;
                color: {color};
            }}

            .confidence {{
                font-size: 18px;
                margin-bottom: 20px;
            }}

            .bar {{
                width: 100%;
                background: rgba(255,255,255,0.2);
                border-radius: 10px;
                overflow: hidden;
                margin: 20px 0;
            }}

            .fill {{
                height: 20px;
                width: {confidence * 100}%;
                background: {color};
                transition: width 0.5s;
            }}

            a {{
                display: inline-block;
                margin-top: 20px;
                padding: 10px 20px;
                background: #333;
                color: white;
                text-decoration: none;
                border-radius: 8px;
            }}

            a:hover {{
                background: #555;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🔍 Analysis Result</h2>

            <div class="result">{label}</div>

            <div class="confidence">
                Confidence: {confidence:.2f}
            </div>

            <div class="bar">
                <div class="fill"></div>
            </div>

            <a href="/">⬅ Try Another</a>
        </div>
    </body>
    </html>
    """


@app.route("/")
def home():
    return render_home()


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        label, confidence = predict_video(filepath)

        return render_result(label, confidence)

    return render_home()


if __name__ == "__main__":
    app.run(debug=True)
