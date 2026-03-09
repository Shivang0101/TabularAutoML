import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s — %(message)s',
    handlers=[
        logging.FileHandler('automl.log'),  # saves every log line to file
        logging.StreamHandler()              # also shows in terminal
    ]
)


import openml
import pandas as pd
from src.pipeline.automl_pipeline import AutoMLPipeline

dataset = openml.datasets.get_dataset(151)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
df = X.copy()
df['credit'] = y

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['credit'] = le.fit_transform(df['credit'])

pipeline = AutoMLPipeline(n_hpo_trials=5, experiment_name='Credit-Demo')
results = pipeline.run(df, target_col='credit')