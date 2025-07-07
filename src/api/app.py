import os
import uuid
import pandas as pd
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename

from src.api.prediction_service import predict_df
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── Flask App Setup ─────────────────────────────────────────────────────────
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PRED_FOLDER = "predictions"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRED_FOLDER, exist_ok=True)

REFERENCE_DATA_PATH = "data/processed/processed_output.csv"
reference_df = pd.read_csv(REFERENCE_DATA_PATH)

MODEL_NAME = "CatBoostRegressor"  # Updated model name in MLflow

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    try:
        # Warmup: load model and run dummy prediction
        _ = predict_df(reference_df.sample(1))
    except Exception as e:
        logger.warning(f"⚠️ Sample prediction failed: {e}")
    return render_template("index.html", model_name=MODEL_NAME)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("index.html", model_name=MODEL_NAME, error="No file uploaded")

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            raw_df = pd.read_csv(file_path)
        except Exception as e:
            return render_template("index.html", model_name=MODEL_NAME, error=f"Invalid file format: {e}")

        try:
            preds = predict_df(raw_df)
            raw_df.drop(columns=["Predicted_Salary", "adjusted_total_salary", "adjusted_total_usd"], errors="ignore", inplace=True)
            raw_df["adjusted_total_usd"] = preds.round(2)  # Rounded prediction
        except Exception as e:
            return render_template("index.html", model_name=MODEL_NAME, error=f"Prediction failed: {e}")

        result_id = str(uuid.uuid4())[:8]
        result_file = f"pred_{result_id}.csv"
        result_path = os.path.join(PRED_FOLDER, result_file)
        raw_df.to_csv(result_path, index=False)

        html_table = raw_df.head(100).to_html(classes="table table-bordered table-striped", index=False)
        return render_template("result.html", tables=[html_table], result_file=result_file, model_name=MODEL_NAME)

    return render_template("index.html", model_name=MODEL_NAME)

@app.route("/sample")
def sample():
    return send_file(REFERENCE_DATA_PATH, as_attachment=True)

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(PRED_FOLDER, filename), as_attachment=True)

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8000)
