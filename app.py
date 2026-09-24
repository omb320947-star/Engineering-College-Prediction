import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load trained models
MODEL_CLG_PATH = "clgname.pkl"
MODEL_COURSE_PATH = "course.pkl"

model_clg = None
model_course = None

if os.path.exists(MODEL_CLG_PATH):
    with open(MODEL_CLG_PATH, "rb") as f:
        model_clg = pickle.load(f)

if os.path.exists(MODEL_COURSE_PATH):
    with open(MODEL_COURSE_PATH, "rb") as f:
        model_course = pickle.load(f)

# Categories and Gender mappings based on standard MHT-CET dataset encodings
# If your model used categorical encoding during training, update these lists to match your exact encoding order
CATEGORY_MAPPING = [
    "GOPEN", "GSCH", "GST", "GNT1", "GNT2", "GNT3", "GOBC",
    "LOPEN", "LSCH", "LST", "LNT1", "LNT2", "LNT3", "LOBC",
    "EWS", "TFWS", "PWD"
]

GENDER_MAPPING = ["Male", "Female"]

# Single-page HTML Template with embedded CSS & JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MHT-CET College & Course Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            color: #f8fafc;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px 0;
        }
        .card-custom {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        .form-control, .form-select {
            background-color: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #fff;
            border-radius: 10px;
            padding: 12px 15px;
        }
        .form-control:focus, .form-select:focus {
            background-color: rgba(15, 23, 42, 0.8);
            border-color: #38bdf8;
            box-shadow: 0 0 0 0.25rem rgba(56, 189, 248, 0.25);
            color: #fff;
        }
        .btn-gradient {
            background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
            border: none;
            color: #fff;
            font-weight: 600;
            padding: 12px;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .btn-gradient:hover {
            background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
            transform: translateY(-2px);
        }
        .result-box {
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 12px;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-6 col-md-8">
                <div class="card card-custom p-4 p-md-5">
                    <h2 class="text-center font-weight-bold mb-2 text-white">College & Course Predictor</h2>
                    <p class="text-center text-muted mb-4 small">Enter your score and details to find your predicted college and engineering branch.</p>

                    {% if error %}
                        <div class="alert alert-danger" role="alert">
                            {{ error }}
                        </div>
                    {% endif %}

                    <form action="/predict" method="POST">
                        <div class="mb-3">
                            <label for="percentile" class="form-label text-light">MHT-CET Percentile</label>
                            <input type="number" step="0.0001" min="0" max="100" class="form-control" id="percentile" name="percentile" placeholder="e.g. 95.50" value="{{ percentile if percentile else '' }}" required>
                        </div>

                        <div class="mb-3">
                            <label for="gender" class="form-label text-light">Gender</label>
                            <select class="form-select" id="gender" name="gender" required>
                                <option value="" disabled {{ 'selected' if not gender else '' }}>Select Gender</option>
                                {% for g in genders %}
                                    <option value="{{ g }}" {{ 'selected' if gender == g else '' }}>{{ g }}</option>
                                {% endfor %}
                            </select>
                        </div>

                        <div class="mb-4">
                            <label for="category" class="form-label text-light">Category</label>
                            <select class="form-select" id="category" name="category" required>
                                <option value="" disabled {{ 'selected' if not category else '' }}>Select Category</option>
                                {% for cat in categories %}
                                    <option value="{{ cat }}" {{ 'selected' if category == cat else '' }}>{{ cat }}</option>
                                {% endfor %}
                            </select>
                        </div>

                        <button type="submit" class="btn btn-gradient w-100">Predict Options</button>
                    </form>

                    {% if predicted_college or predicted_course %}
                        <div class="result-box mt-4">
                            <h5 class="text-info font-weight-bold mb-3">Prediction Results</h5>
                            {% if predicted_college %}
                                <div class="mb-2">
                                    <small class="text-muted d-block">Predicted College:</small>
                                    <strong class="text-white fs-6">{{ predicted_college }}</strong>
                                </div>
                            {% endif %}
                            {% if predicted_course %}
                                <div>
                                    <small class="text-muted d-block">Predicted Course/Branch:</small>
                                    <strong class="text-white fs-6">{{ predicted_course }}</strong>
                                </div>
                            {% endif %}
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        HTML_TEMPLATE,
        genders=GENDER_MAPPING,
        categories=CATEGORY_MAPPING
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        percentile = float(request.form.get("percentile", 0))
        gender_str = request.form.get("gender")
        category_str = request.form.get("category")

        # Encode categorical variables as numbers for feature array
        gender_encoded = GENDER_MAPPING.index(gender_str) if gender_str in GENDER_MAPPING else 0
        category_encoded = CATEGORY_MAPPING.index(category_str) if category_str in CATEGORY_MAPPING else 0

        # Feature order expected by model: [Percentile, Gender, Category]
        features = np.array([[percentile, gender_encoded, category_encoded]])

        predicted_college = None
        predicted_course = None

        if model_clg is not None:
            pred_clg = model_clg.predict(features)[0]
            predicted_college = str(pred_clg)

        if model_course is not None:
            pred_crs = model_course.predict(features)[0]
            predicted_course = str(pred_crs)

        return render_template_string(
            HTML_TEMPLATE,
            genders=GENDER_MAPPING,
            categories=CATEGORY_MAPPING,
            percentile=percentile,
            gender=gender_str,
            category=category_str,
            predicted_college=predicted_college,
            predicted_course=predicted_course
        )

    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            genders=GENDER_MAPPING,
            categories=CATEGORY_MAPPING,
            error=f"Error in prediction: {str(e)}"
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
