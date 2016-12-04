import csv
import pandas as pd
import sys

"""
with open('predictions.csv', 'a') as csvfile:
        testwriter = csv.writer(csvfile, delimiter=',')
"""
submission = pd.read_csv(sys.argv[1])
submission.columns = ["File", "Class"]
submission = submission[submission["File"] != "File"]
for i in submission.index:
    x = submission["Class"][i]
    x = x.replace("[","")
    x = x.replace("]","")
    val = float(x.split()[1])
    submission["Class"][i] = val
submission.to_csv("submission_fixed.csv",index=False)
print(submission)
