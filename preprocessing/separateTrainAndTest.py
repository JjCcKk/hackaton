import csv
import pandas as pd
import os
import random


with open("fullDatasetWithLables.csv","r") as file:
    with open("train.csv",'a') as train:
        trainWriter = csv.writer(train)
        with open("test.csv","a") as test:
            testWriter = csv.writer(test)
            for i in file.readlines():
                print(i)
                r = random.random()
                if r <= 0.2:
                    testWriter.writerow([str(i)])
                else:
                    trainWriter.writerow([str(i)])
            test.close()
        train.close()
    file.close()


