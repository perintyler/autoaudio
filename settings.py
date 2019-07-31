from dotenv import load_dotenv
import os

load_dotenv()

# set api keys from .env file: os.getenv('<VARIABLE_NAME>')
classification_dir = 'models'
training_data_dir = 'training_data'
feature_dir = 'features'
prediction_dir = 'predictions'


label_indecies = {
    'breathing': 0,
    'coughing': 1
}
