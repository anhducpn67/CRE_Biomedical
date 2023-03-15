import os
import numpy as np


dir = "../data_csv"
file_list = ["ADE_train.csv","ADE_valid.csv","ADE_test.csv",]

for file in file_list:
    file_path = os.path.join(dir, file)
    with open(file_path, "r") as f:
        data = f.readlines()

    len_list = []
    for data_item in data:
        data_item_list = data_item.split("||")
        text_len = len(data_item_list[1].split(" "))
        len_list.append(text_len)

    print("average len", np.average(len_list))
    print("max len", np.max(len_list))
    print("min len", np.min(len_list))
