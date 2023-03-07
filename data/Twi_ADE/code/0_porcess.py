import os
import numpy as np


def parse_file(raw_data_dir, file_path):
    text_file = os.path.join(raw_data_dir, file_path+ ".txt")
    ann_file = os.path.join(raw_data_dir, file_path+ ".ann")
    pub_med_ID = file_path[3:]
    with open(text_file, "r") as f:
        text = f.readlines()

    with open(ann_file, "r") as f:
        anno = f.readlines()

    # [('C055162', 'clopidogrel', (32, 43), 'Chemical'), ('D056486', 'hepatotoxicity', (102, 116), 'Disease'), 'CID']
    entity_dic = {}
    relation_list = []
    existed_entity_list = []
    existed_relation_list = []
    anno_list = []

    for anno_item in anno:
        anno_item_list = anno_item.split("\t")
        if anno_item_list[0][0] == "T":
            temp_entity_list = anno_item_list[1].split(" ")
            try:
                assert len(temp_entity_list)==3
                temp_entity_list_list = [temp_entity_list]
            except:
                assert len(temp_entity_list)==4

                temp_entity_list_list = [[temp_entity_list[0], temp_entity_list[1],  temp_entity_list[2].split(";")[0]],
                                         [temp_entity_list[0], temp_entity_list[2].split(";")[1], temp_entity_list[3]]
                                         ]

            for index, temp_entity_list in enumerate(temp_entity_list_list):
                entity_type, entity_S, entity_E = temp_entity_list
                for i in range(int(entity_S), int(entity_E)):
                    existed_entity_list.append(i)
                entity_text = anno_item_list[2].replace("\n", "")
                if entity_S not in existed_entity_list and entity_E not in existed_entity_list:
                    if index>0:
                        temp_key = anno_item_list[0]+"_"+str(index)
                    else:
                        temp_key = anno_item_list[0]
                    entity_dic[temp_key] = (anno_item_list[0], entity_text, (entity_S, entity_E), entity_type )
                else:
                    print("neseted entity !")
                    print(entity_S, entity_E)

        if anno_item_list[0][0] == "R":
            # Benefit Arg1:T2 Arg2:T3
            temp_relation_list = anno_item_list[1].split(" ")
            assert len(temp_entity_list)==3
            relation_type, entity_1, entity_2 = temp_relation_list
            entity_1 = entity_1.replace("Arg1:", "").replace("\n", "")
            entity_2 = entity_2.replace("Arg2:", "").replace("\n", "")
            assert entity_1 in entity_dic.keys()
            assert entity_2 in entity_dic.keys()
            existed_relation_list.append(entity_1)
            existed_relation_list.append(entity_2)
            if int(entity_dic[entity_1][2][0]) < int(entity_dic[entity_2][2][0]):
                relation_list.append([entity_dic[entity_1], entity_dic[entity_2], relation_type])
            else:
                relation_list.append([entity_dic[entity_2], entity_dic[entity_1], relation_type])

    new_entity_dic= entity_dic.copy()
    for k in entity_dic.keys():
        if k in existed_relation_list:
            del new_entity_dic[k]
        if len(k.split("_"))>1 and (k.split("_")[0] in existed_relation_list):
            del new_entity_dic[k]

    anno_list.extend(relation_list)
    for i in list(new_entity_dic.values()):
        anno_list.append([i])

    return_list = [str(pub_med_ID), str(text[0]), str(anno_list)]
    return "||".join(return_list)

def process_0(raw_data_dir, res_file):

    file_path = []
    for file_triple in os.walk(raw_data_dir):

        for file in file_triple[2]:
            file = file.split(".")[0]
            if file not in file_path:
                file_path.append(file)
    assert len(file_path) == 1000

    with open(res_file, "w") as f:
        for file in file_path:
            temp_str = parse_file(raw_data_dir, file)
            f.writelines(temp_str+"\n")

def process_1(res_file):

    with open(res_file, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()

    test_valid_num = int( len(raw_data) *0.1 )
    train_index = list(range(len(raw_data)))
    valid_index = []
    test_index = []

    for i in range(test_valid_num):
        valid_randIndex = int(np.random.uniform(0, len(train_index) )) # 获得0~len(trainingSet)的一个随机数
        valid_index.append(train_index[valid_randIndex])
        del(train_index[valid_randIndex])

        test_randIndex = int(np.random.uniform(0, len(train_index) )) # 获得0~len(trainingSet)的一个随机数
        test_index.append(train_index[test_randIndex])
        del(train_index[test_randIndex])


    file_list = ["Twi_ADE_train.csv", "Twi_ADE_valid.csv", "Twi_ADE_test.csv"]
    index_list = [train_index, valid_index, test_index]

    for index, file in enumerate(file_list):
        file = os.path.join('../data_csv', file)
        with open(file, 'w') as f:
            for data_index in index_list[index]:
                data = raw_data[data_index]
                f.writelines(data)

if __name__ == '__main__':
    raw_data_dir= '../raw_data/'
    res_file = '../data_csv/Twi_ADE.csv'
    process_0(raw_data_dir, res_file)
    process_1(res_file)


