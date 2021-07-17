import torch

IEMOCAP_JSON_PATH = 'D:/PythonProject/IEMOCAP_Data/IEMOCAP_DATA/'
IEMOCAP_SAVE_PATH = 'D:/PythonProject/IEMOCAP_ExperimentResult/'
# IEMOCAP_JSON_PATH = '/mnt/external/Bobs/IEMOCAP_DATA'
# IEMOCAP_SAVE_PATH = 'Result/'

AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AVAILABLE_GPU_NUMBER = torch.cuda.device_count()
ADAM_EPSILON = 1E-8
LEARNING_RATE = 5E-5
MAX_GRAD_NORM = 1.0

CLASS_NUMBER = 4
EMOTION_LABEL = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
FEATURE_SHAPE = {'Audio': 40, 'Video': 168}
