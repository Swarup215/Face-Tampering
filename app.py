from flask import Flask, render_template, request
import os
from predict import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def home():
    image_path = None
    label = None
    confidence = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(save_path)

            label, confidence = predict_image(save_path)
            image_path = save_path

    return render_template(
        "index.html",
        image_path=image_path,
        label=label,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)