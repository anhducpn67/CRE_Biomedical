import os
import numpy as np


dir = "../data_csv"
file_list = ["Test_DrugBank.csv","Test_MedLine.csv","Train_DrugBank.csv","Train_MedLine.csv"]

drug_n_count = 0
drug_n_relation_count = 0
for file in file_list:
    print(file)
    file_path = os.path.join(dir, file)
    with open(file_path, "r") as f:
        data = f.readlines()

    len_list = []

    for data_item in data:
        data_item_list = data_item.split("||")
        text_len = len(data_item_list[1].split(" "))
        len_list.append(text_len)
        if len(eval(data_item_list[2]))>1:

            e1 = eval(data_item_list[2])[0][-1][-1]
            e2 = eval(data_item_list[2])[1][-1][-1]
            if  e1== "drug_n":
                drug_n_count+=1
            if e2  == "drug_n":
                drug_n_count+=1
            if e1== "drug_n" or e2  == "drug_n":
                drug_n_relation_count+=1
        else:
            if eval(data_item_list[2])[0][-1][-1] == "drug_n":
                drug_n_count+=1


    print("average len", np.average(len_list))
    print("max len", np.max(len_list))
    print("min len", np.min(len_list))


print("drug_n_count", drug_n_count)
print("drug_n_relation_count", drug_n_relation_count)