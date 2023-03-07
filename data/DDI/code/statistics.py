import numpy as np
from itertools import combinations


def statistics_corpus(train_file):

    def weight_fn(raw_dic):
        total_log_num = 0
        for k,v in raw_dic.items():
            total_log_num += np.log(v)
        ratio_relation_dic_1 = raw_dic.copy()
        ratio_relation_dic_1.update((k,round(np.log(total_log_num/np.log(v)),4)) for k,v in ratio_relation_dic_1.items())
        return ratio_relation_dic_1

    def weight_fn_1(raw_dic):
        return_dic = {}
        for k,v in raw_dic.items():
            total_log_num = np.sum(np.log(v))
            dic_temp = {}
            for index, i in enumerate(v) :
                dic_temp[index] = round(np.log(total_log_num/np.log(i)), 4)
            return_dic[k] = dic_temp
        return return_dic

    with open(train_file, "r") as f:
        train_data = f.readlines()

    relation_list = ["relation_"+str(i) for i in ["mechanism", "effect", "advise", "int"]]
    num_relation_dic = {}
    for relation in relation_list:
        num_relation_dic.setdefault(relation, 0)

    total_relation_num = 0
    total_entitiy_pair_num = 0
    for i in train_data:
        data_dic = eval(i)
        entity_combine_len = len(list(combinations(data_dic["sep_entity"], 2)))
        total_entitiy_pair_num += entity_combine_len
        for k, v in data_dic.items():
            if k in relation_list:
                num_relation_dic[k] = num_relation_dic[k] + len(v)
                total_relation_num+= len(v)

    ratio_relation_dic = num_relation_dic.copy()
    ratio_relation_dic.update((k,round(v/total_relation_num,4)) for k,v in ratio_relation_dic.items())
    ratio_relation_list = list(ratio_relation_dic.values())

    ratio_relation_dic_1 = weight_fn(num_relation_dic)

    raw_yes_no_relation_dic = {}
    for k,v in num_relation_dic.items():
        raw_yes_no_relation_dic[k] = [v, total_entitiy_pair_num-v]


    yes_no_relation_dic = num_relation_dic.copy()
    yes_no_relation_dic.update((k,round(v/total_entitiy_pair_num,8)) for k,v in yes_no_relation_dic.items())
    yes_no_relation_list = [[i, 1-i] for i in list(yes_no_relation_dic.values())]
    # yes_no_relation_list_1 = [[np.log(1/i)] for i in list(yes_no_relation_dic.values())]
    yes_no_relation_list_1 = [[1/i] for i in list(yes_no_relation_dic.values())]

    yes_no_relation_dic_2 = weight_fn_1(raw_yes_no_relation_dic)

    print("total_relation_num", total_relation_num)
    print("total_entitiy_pair_num", total_entitiy_pair_num)
    print("num_relation_dic", num_relation_dic)
    print("ratio_relation_dic", ratio_relation_dic)
    print("ratio_relation_list", ratio_relation_list)
    print("ratio_relation_list", [np.log(1/i) for i in ratio_relation_list])
    print("ratio_relation_list_1", list(ratio_relation_dic_1.values()))

    print("raw_yes_no_relation_dic", raw_yes_no_relation_dic)
    print("yes_no_relation_dic", yes_no_relation_dic)
    print("yes_no_relation_list", yes_no_relation_list)
    print("yes_no_relation_list_1", yes_no_relation_list_1)
    print("yes_no_relation_dic_2", yes_no_relation_dic_2)

    return ratio_relation_list, yes_no_relation_list





train_file = "../BIOES/base/DDI_train_base_model_data.json"
# valid_file = "./CPR_valid_base_model_data.json"
# test_file = "./CPR_test_base_model_data.json"
# file_list = [train_file, valid_file, test_file]
file_list = [train_file]
for file in file_list:
    _, _ = statistics_corpus(file)
    print()

