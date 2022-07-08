import pandas as pd
from config import ANNOTATIONS_FILE_TRAIN, ANNOTATIONS_FILE_TEST, DATA_INI, JSON_DATA

if __name__=="__main__":
    print(JSON_DATA)
    data = pd.read_json(JSON_DATA).T
    data["Path"] = [DATA_INI + str(i)[0] + "_" + str(i)[1:] + ".mp4" for i in data.index]
    data.index = range(data.shape[0])
    train = data.loc[:int(data.shape[0]*0.8)].to_csv(ANNOTATIONS_FILE_TRAIN, index=False)
    test = data.loc[int(data.shape[0]*0.8):].to_csv(ANNOTATIONS_FILE_TEST, index=False)
