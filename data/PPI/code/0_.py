import os

import numpy as np
from lxml import etree


def delete_relation_item(temp_str_list, entity_1, list_relation_len):
    for item in temp_str_list[:list_relation_len]:
        entity_pair_comp = (item[0], item[1])
        for entity_2 in entity_pair_comp:
            if entity_1 != entity_2 and ((entity_1[2][0] >= entity_2[2][0] and entity_1[2][1] <= entity_2[2][1])
                                         or (entity_2[2][0] <= entity_1[2][0] <= entity_2[2][1] and len(
                        entity_1[1]) <= len(entity_2[1]))
                                         or (entity_2[2][0] <= entity_1[2][1] <= entity_2[2][1] and len(
                        entity_1[1]) <= len(entity_2[1]))):
                return True

    return False


def delete_entity_item(temp_str_list, entity_1, list_relation_len):
    for another_index, item in enumerate(temp_str_list[:list_relation_len]):
        entity_pair_comp = (item[0], item[1])
        for entity_2 in entity_pair_comp:
            if entity_1 != entity_2 and ((entity_1[2][0] >= entity_2[2][0] and entity_1[2][1] <= entity_2[2][1])
                                         or (entity_1[2][0] <= entity_2[2][0] and entity_1[2][1] >= entity_2[2][1])
                                         or (entity_2[2][0] <= entity_1[2][0] <= entity_2[2][1])
                                         or (entity_2[2][0] <= entity_1[2][1] <= entity_2[2][1])):
                return True
    for another_index, item in enumerate(temp_str_list[list_relation_len:]):
        entity_2 = item[0]
        if entity_1 != entity_2 and ((entity_1[2][0] >= entity_2[2][0] and entity_1[2][1] <= entity_2[2][1])
                                     or (entity_2[2][0] <= entity_1[2][0] <= entity_2[2][1] and len(entity_1[1]) <= len(
                    entity_2[1]))
                                     or (entity_2[2][0] <= entity_1[2][1] <= entity_2[2][1] and len(entity_1[1]) <= len(
                    entity_2[1]))):
            return True
    return False


def process_0(xml, f, type1, type2):
    tree = etree.parse(xml)
    corpus = tree.getroot()
    count_pair = 0
    sentence_count = 0
    for document in corpus:
        document_id = document.attrib['id']
        for sentence in document:
            sentence_id = sentence.attrib['id']
            sentence_text = sentence.attrib['text'].replace('\n', '').replace('\r', '')
            entity_dict = {}
            entity_non_relation = {}
            for entity_or_pair in sentence:
                if entity_or_pair.tag == 'entity':
                    entity = entity_or_pair
                    entity_id = entity.attrib['id']
                    entity_text = entity.attrib['text']
                    entity_type = entity.attrib['type']
                    if entity_type not in type1:
                        type1.append(entity_type)
                    entity_start, entity_end = entity.attrib['charOffset'].split(';')[0].split('-')
                    entity_dict[entity_id] = (
                    entity_id, entity_text, (int(entity_start), int(entity_end) + 1), entity_type)
                    entity_non_relation[entity_id] = [
                        (entity_id, entity_text, (int(entity_start), int(entity_end) + 1), entity_type)]
            pair_list = []
            for entity_or_pair in sentence:
                if entity_or_pair.tag == 'pair':
                    count_pair += 1
                    if entity_or_pair.attrib['interaction'] == 'True':
                        pair = entity_or_pair
                        pair_id = pair.attrib['id']
                        pair_type = "Gene_Gene_interaction"
                        pair_subj_id = pair.attrib['e1']
                        pair_obj_id = pair.attrib['e2']
                        if pair_type not in type2:
                            type2.append(pair_type)
                        if pair_subj_id in entity_non_relation.keys():
                            entity_non_relation.pop(pair_subj_id)
                        if pair_obj_id in entity_non_relation.keys():
                            entity_non_relation.pop(pair_obj_id)

                        pair_list.append([entity_dict[pair_subj_id], entity_dict[pair_obj_id], pair_type, pair_id])
            annotations = pair_list
            list_relation_len = len(pair_list)
            annotations.extend(list(entity_non_relation.values()))

            delete_item_list = []
            delete_item_dict = {}
            for index, relation_item in enumerate(annotations[:list_relation_len]):
                delete_flag_entity_1 = delete_relation_item(annotations, relation_item[0], list_relation_len)
                delete_flag_entity_2 = delete_relation_item(annotations, relation_item[1], list_relation_len)
                if delete_flag_entity_1 and delete_flag_entity_2:
                    delete_item_dict[str(relation_item)] = 2
                elif delete_flag_entity_1:
                    delete_item_dict[str(relation_item)] = 1
                elif delete_flag_entity_2:
                    delete_item_dict[str(relation_item)] = 0
            for index, entity_1 in enumerate(annotations[list_relation_len:]):
                delete_flag = delete_entity_item(annotations, entity_1[0], list_relation_len)
                if delete_flag:
                    delete_item_list.append(entity_1)

            # print(delete_item_dict, delete_item_list)
            for item, num in delete_item_dict.items():
                item = eval(item)
                annotations.remove(item)
                if num != 2:
                    annotations.append([item[num]])
            for item in delete_item_list:
                annotations.remove(item)

            if annotations:
                sentence_count += 1
                f.write(f'55555{sentence_count}||{sentence_text}||{str(annotations)}\n')
    print("Number of pairs:", count_pair)
    return type1, type2


def process_1(res_file):
    with open(res_file, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()

    test_valid_num = int(len(raw_data) * 0.1)
    train_index = list(range(len(raw_data)))
    valid_index = []
    test_index = []

    for i in range(test_valid_num):
        valid_randIndex = int(np.random.uniform(0, len(train_index)))  # 获得0~len(trainingSet)的一个随机数
        valid_index.append(train_index[valid_randIndex])
        del (train_index[valid_randIndex])

        test_randIndex = int(np.random.uniform(0, len(train_index)))  # 获得0~len(trainingSet)的一个随机数
        test_index.append(train_index[test_randIndex])
        del (train_index[test_randIndex])

    file_list = ["PPI_train.csv", "PPI_valid.csv", "PPI_test.csv"]
    index_list = [train_index, valid_index, test_index]

    for index, file in enumerate(file_list):
        file = os.path.join('../data_csv', file)
        with open(file, 'w') as f:
            for data_index in index_list[index]:
                data = raw_data[data_index]
                f.writelines(data)


if __name__ == '__main__':
    raw_data = '../raw_data/AImed.xml'
    res_file = '../data_csv/PPI.csv'
    entity_type = []
    relation_type = []
    with open(res_file, "w", encoding='utf-8') as f:
        process_0(raw_data, f, entity_type, relation_type)
    print('entity_type =', entity_type)
    print('relation_type =', relation_type)
    process_1(res_file)
