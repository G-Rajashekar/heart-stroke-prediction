from flask import Flask, request
import sys
import pip
from heart_stroke.utils.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from heart_stroke.logger import logging
import os, sys
import json
from heart_stroke.config.configuration import Configuartion
from heart_stroke.constant import CONFIG_DIR, get_current_time_stamp
from heart_stroke.pipeline.training_pipeline import Pipeline
from heart_stroke.entity.heart_stroke_predictor import predictor, HeartStrokeData
from flask import send_file, abort, render_template

ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "heart_stroke"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)

from heart_stroke.logger import get_log_dataframe

MODEL_DATA_KEY = "heart_stroke_data"
PREDICTED_CLASS_KEY = "stroke"
PROBABILITY_KEY = "proba"

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'heart_stroke'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("heart_stroke", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuartion(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        MODEL_DATA_KEY: None,
        PREDICTED_CLASS_KEY: None,
        PROBABILITY_KEY: None
    }
    if request.method == 'POST':
        try:
            gender = request.form.get('gender')
            age_raw = request.form.get('age')
            hypertension_raw = request.form.get('hypertension')
            heart_disease_raw = request.form.get('heart_disease')
            ever_married = request.form.get('ever_married')
            work_type = request.form.get('work_type')
            Residence_type = request.form.get('Residence_type')
            avg_glucose_level_raw = request.form.get('avg_glucose_level')
            smoking_status = request.form.get('smoking_status')
            bmi_raw = request.form.get('bmi')

            # Basic validation for required fields
            required_values = [gender, age_raw, hypertension_raw, heart_disease_raw,
                               ever_married, work_type, Residence_type,
                               avg_glucose_level_raw, smoking_status, bmi_raw]
            if any(v in (None, "") for v in required_values):
                context[PREDICTED_CLASS_KEY] = "Please fill in all fields before predicting."
                return render_template('predict.html', context=context)

            age = int(age_raw)
            hypertension = int(hypertension_raw)
            heart_disease = int(heart_disease_raw)
            avg_glucose_level = float(avg_glucose_level_raw)
            bmi = float(bmi_raw)

            # Ensure model exists and contains at least one numeric version folder
            if not os.path.isdir(MODEL_DIR):
                context[PREDICTED_CLASS_KEY] = "Model is not available. Please train the model first from the Train page."
                return render_template('predict.html', context=context)

            try:
                numeric_versions = [int(name) for name in os.listdir(MODEL_DIR) if name.isdigit()]
            except Exception:
                numeric_versions = []
            if len(numeric_versions) == 0:
                context[PREDICTED_CLASS_KEY] = "No trained model found. Please run training from the Train page."
                return render_template('predict.html', context=context)

            heart_stroke_data = HeartStrokeData(
                gender=gender,
                age=age,
                hypertension=hypertension,
                heart_disease=heart_disease,
                ever_married=ever_married,
                work_type=work_type,
                Residence_type=Residence_type,
                avg_glucose_level=avg_glucose_level,
                bmi=bmi,
                smoking_status=smoking_status
            )

            heart_stroke_df = heart_stroke_data.get_heart_stroke_input_data_frame()
            stroke_predictor = predictor(model_dir=MODEL_DIR)
            # Extra guard: ensure latest model path is resolvable
            try:
                _ = stroke_predictor.get_latest_model_path()
            except Exception:
                context[PREDICTED_CLASS_KEY] = "Unable to load the trained model. Please retrain from the Train page."
                return render_template('predict.html', context=context)

            predicted_value = stroke_predictor.predict(X=heart_stroke_df)[0]
            if predicted_value == 1:
                predicted_message = "The Patient has high chance of Heart Stroke"
            else:
                predicted_message = "The Patient has No chance of Heart Stroke"

            probability_val = stroke_predictor.proba_predict(X=heart_stroke_df)[0][1]
            probability_text = f'Probability of stroke is {probability_val * 100:.2f}%'
            context = {
                MODEL_DATA_KEY: heart_stroke_data.get_heart_stroke_data_as_dict(),
                PREDICTED_CLASS_KEY: predicted_message,
                PROBABILITY_KEY: probability_text
            }

            return render_template('predict.html', context=context)
        except Exception as e:
            logging.exception(e)
            context[PREDICTED_CLASS_KEY] = "An error occurred while predicting. Please verify inputs or train the model."
            return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/bmi', methods=['GET', 'POST'])
def bmi():
    result = None
    error = None
    if request.method == 'POST':
        try:
            height_cm = float(request.form.get('height_cm', '').strip())
            weight_kg = float(request.form.get('weight_kg', '').strip())
            if height_cm <= 0 or weight_kg <= 0:
                raise ValueError("Non-positive")
            height_m = height_cm / 100.0
            bmi_val = weight_kg / (height_m * height_m)
            category = (
                "Underweight" if bmi_val < 18.5 else
                "Normal" if bmi_val < 25 else
                "Overweight" if bmi_val < 30 else
                "Obese"
            )
            result = {"bmi": f"{bmi_val:.2f}", "category": category}
        except Exception:
            error = "Please enter valid height (cm) and weight (kg)."
    return render_template('bmi.html', result=result, error=error)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)