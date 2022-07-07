import csv
import os
#import os.path

# open the file in the write mode
with open('annotations.csv', 'a') as f:
    # create the csv writer
    writer = csv.writer(f,lineterminator='\n')




    rootdir = os.getcwd()

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = os.path.join(subdir, file)
            filename, file_extension = os.path.splitext(str(filepath))
            #filepath = subdir + os.sep + file
            print(filepath)
            #print(file)
            print(len(filepath))
            #write name and path to new line
            print(file_extension) ###gogogogoogogog if blah 
            writer.writerow([filepath])



# close the csv file
f.close()
