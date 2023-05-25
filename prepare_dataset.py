import pandas as pd
import os
 
# 
# Command and Control Words-W1
# Form and Function Word-W5
# Most Frequent Word-FullSet-W3B
# Most Frequent Word-Part-W3A
# Person Name-W2
# Phonetically Balanced-W4
# Place Name-W2
import re
def get_file_contents(file_path):
    with open(file_path) as input_file:
        data = input_file.read().rstrip()
        return (re.search('RECORDED TEXT :: \n\n(.*)?',data).group(1))
if __name__ == "__main__":
    print("1.Start")
    # Get the list of all files and directories
    # in the root directory
    path = "./KonkaniRawSpeechCorpus/Data"
    dir_list = os.listdir(path)
    data  = []
    print("Files and directories in '", path, "' :") 
    gender_list =["Male","Female"]
    age_list = ["16To20","21To50","Above51"]
    # print the list
    for dir in dir_list:
        for gender in gender_list:
            for age in age_list:
                input_data_dir_path = os.path.join(path,dir,gender,age)
                print(input_data_dir_path)
                for txt_file in os.listdir(input_data_dir_path):
                    if txt_file.endswith(".txt"):
                        data_file_path = os.path.join(input_data_dir_path, txt_file)
                        print(data_file_path)
                        data.append({"sentence":get_file_contents(data_file_path),
                                     "txt_path":data_file_path})
    pd.DataFrame.from_dict(data).to_csv("KonkaniCorpusDataset.csv",sep="\t",index=False)
    print("2.End")