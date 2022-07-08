import csv
import pandas as pd
import os
#import os.path

# open the file in the write mode
with open('annotations.csv', 'a') as f:
    # create the csv writer
    writer = csv.writer(f,lineterminator='\n')

    jfile = pd.read_json("sarcasm_data.json")
    jfile = pd.DataFrame(jfile)


    rootdir = os.getcwd()

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = os.path.join(subdir, file)
            filename, file_extension = os.path.splitext(str(filepath))
            filename = filename.split("/")[-1]
            key = filename.replace("_","")
            print(key)
            if key.split()[0].isnumeric(): 
                #Read JSON file and extract sarcasm-lable
                lable = jfile[int(key)]['sarcasm']
                print(f"key:{key} lable:{lable}")
                line = filepath + "," + filename + "," + str(lable)
                writer.writerow([filepath])
            else:
                print("Non audio file skipped")



# close the csv file
f.close()
