import pandas as pd
import os
from config import ANNOTATIONS_FILE_TRAIN, ANNOTATIONS_FILE_TEST, DATA_INI, JSON_DATA


def create_archi():
    if not os.path.exists("data"):
        os.makedirs("data")

def segment_data(json_data, data_ini):
    data = pd.read_json(JSON_DATA).T
    data["Path"] = [DATA_INI + str(i)[0] + "_" + str(i)[1:] + "_c.mp4" for i in data.index]
    data.index = range(data.shape[0])

    create_archi()
    train = data.loc[:int(data.shape[0]*0.8)].to_csv(ANNOTATIONS_FILE_TRAIN, index=False)
    test = data.loc[int(data.shape[0]*0.8):].to_csv(ANNOTATIONS_FILE_TEST, index=False)

    return (train, test)

if __name__=="__main__":
    segment_data(JSON_DATA, DATA_INI)