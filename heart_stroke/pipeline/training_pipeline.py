import pandas as pd

class Pipeline:
    experiment = type("experiment", (), {"running_status": False})()

    def __init__(self, config):
        self.config = config

    def start(self):
        print("Pipeline started...")
        Pipeline.experiment.running_status = True
        # Add training logic here
        Pipeline.experiment.running_status = False

    @staticmethod
    def get_experiments_status():
        return pd.DataFrame([{
            "Timestamp": "2025-07-17 15:00",
            "Status": "Completed"
        }])
