import os
import time
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from heart_stroke.utils.util import save_object
from heart_stroke.logger import logging

class Pipeline:
    experiment = type("experiment", (), {"running_status": False})()

    def __init__(self, config):
        self.config = config

    def start(self):
        print("Pipeline started...")
        Pipeline.experiment.running_status = True
        try:
            # Resolve dataset path (ENV override -> project path -> user's Downloads)
            env_path = os.environ.get("HEART_STROKE_DATA_PATH")
            candidates = []
            if env_path:
                candidates.append(env_path)
            candidates.append(os.path.join("NoteBooks", "data", "healthcare-dataset-stroke-data.csv"))
            candidates.append(os.path.expanduser(os.path.join("~", "Downloads", "healthcare-dataset-stroke-data.csv")))

            data_path = None
            for p in candidates:
                if p and os.path.isfile(p):
                    data_path = p
                    break
            if data_path is None:
                raise FileNotFoundError("Dataset not found. Set HEART_STROKE_DATA_PATH or place CSV under NoteBooks/data or ~/Downloads.")

            df = pd.read_csv(data_path)

            # Required columns
            feature_cols = [
                "gender", "age", "hypertension", "heart_disease", "ever_married",
                "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"
            ]
            target_col = "stroke"

            # Drop rows with missing essentials
            df = df[feature_cols + [target_col]].dropna()

            X = df[feature_cols]
            y = df[target_col].astype(int)

            categorical = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
            numerical = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                    ("num", StandardScaler(), numerical),
                ]
            )

            clf = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=42)
            model = SkPipeline(steps=[("pre", preprocessor), ("clf", clf)])

            model.fit(X, y)

            # Save model under saved_models/<numeric_version>/model.pkl
            version = str(int(time.time()))
            export_dir = os.path.join("saved_models", version)
            os.makedirs(export_dir, exist_ok=True)
            model_path = os.path.join(export_dir, "model.pkl")
            save_object(file_path=model_path, obj=model)
            logging.info(f"Model trained and saved to {model_path}")
        except Exception as e:
            logging.exception(e)
        finally:
            Pipeline.experiment.running_status = False

    @staticmethod
    def get_experiments_status():
        return pd.DataFrame([{
            "Timestamp": "2025-07-17 15:00",
            "Status": "Completed"
        }])
