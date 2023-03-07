import os
import numpy as np


def test_1():
    dir = "../data_csv"
    file_list = ["CPR_train.csv", "CPR_valid.csv", "CPR_test.csv"]

    Total_num=0
    relation_num=0

    CHEMICAL_num = 0
    Gene_num = 0

    for file in file_list:
        file_path = os.path.join(dir, file)
        with open(file_path, "r") as f:
            data = f.readlines()
        Total_num+=len(data)

        len_list = []
        for data_item in data:
            data_item_list = data_item.split("||")
            text_len = len(data_item_list[1].split(" "))
            len_list.append(text_len)

            anno_list = eval(data_item_list[2])
            temp_dic = {"GENE":[], "CHEMICAL":[]}
            for anno in anno_list:
                if len(anno) ==5:
                    relation_num +=1

                    if anno[0][-1] == 'GENE-Y' or anno[0][-1] =='GENE-N':
                        temp_dic["GENE"].append(anno[0][-2])
                    elif anno[0][-1] == 'CHEMICAL':
                        temp_dic["CHEMICAL"].append(anno[0][-2])

                    if anno[1][-1] == 'GENE-Y' or anno[1][-1]=='GENE-N':
                        temp_dic["GENE"].append(anno[1][-2])
                    elif anno[1][-1] == 'CHEMICAL':
                        temp_dic["CHEMICAL"].append(anno[1][-2])
                elif len(anno) ==1:
                    if anno[0][-1] == 'GENE-Y' or anno[0][-1]=='GENE-N':
                        temp_dic["GENE"].append(anno[0][-2])
                    elif anno[0][-1] == 'CHEMICAL':
                        temp_dic["CHEMICAL"].append(anno[0][-2])
                else:
                    raise Exception("~")
            CHEMICAL_num += len(set(temp_dic["CHEMICAL"]))
            Gene_num += len(set(temp_dic["GENE"]))

    print(Total_num)
    print(relation_num)

    print(CHEMICAL_num)
    print(Gene_num)

def test_2():
    dir = "../BIOES/base"
    file_list = ["CPR_train_base_model_data.json", "CPR_valid_base_model_data.json", "CPR_test_base_model_data.json"]


    # 'GENE-Y'
    # 'GENE-N'
    # 'CHEMICAL'

    for file in file_list:

        Total_num=0
        relation_num=0
        CHEMICAL_num = 0
        Gene_num = 0


        file_path = os.path.join(dir, file)
        with open(file_path, "r") as f:
            data = f.readlines()

        Total_num +=len(data)

        for data_item in data:
            anno_dic = eval(data_item)
            relation_num+=len(anno_dic["relation_Drug_Gene_interaction"])
            CHEMICAL_num+=len(anno_dic["only_entity_type_Drug"])
            Gene_num+=len(anno_dic["only_entity_type_Gene"])


        print(file)
        print(Total_num)
        print(CHEMICAL_num)
        print(Gene_num)
        print(relation_num)

# test_1()
print()
test_2()

# dir_new = "../BIOES/base/"
# file_list_new = ["CPR_test_base_model_data.json"]
# for file in file_list_new:
#     file_path = os.path.join(dir_new, file)
#     with open(file_path, "r") as f:
#         data = f.readlines()
#
#     len_list = []
#     for data_item in data:
#         token_list = eval(data_item)["tokens"]
#         text_len = len(token_list)
#         len_list.append(text_len)
#
#     print("new average len", np.average(len_list))
#     print("new max len", np.max(len_list))
#     print("new min len", np.min(len_list))