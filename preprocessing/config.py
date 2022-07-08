import os

#Preprocessing configuration

DATA_INI = os.getcwd() + "/mmsd_raw_data/utterances_final"
JSON_DATA = os.getcwd() + "/MUStARD/data/sarcasm_data.json"
ANNOTATIONS_FILE_TRAIN = "train.csv"
ANNOTATIONS_FILE_TEST = "test.csv"
AUDIO_DIR = "/"
SAMPLE_RATE = 44100
#NUM_SAMPLES = 44100
NUM_SAMPLES = 44100*5