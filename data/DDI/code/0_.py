import os
from lxml import etree

def delete_relation_item(temp_str_list, entity_1, list_relation_len):
    for item in temp_str_list[:list_relation_len]:
        entity_pair_comp = (item[0], item[1])
        for entity_2 in entity_pair_comp:
            if entity_1 != entity_2 and ((entity_1[2][0] >= entity_2[2][0] and entity_1[2][1] <= entity_2[2][1])
                                         or (entity_1[2][0] >= entity_2[2][0] and entity_1[2][0] <= entity_2[2][1] and len(entity_1[1]) <= len(entity_2[1]))
                                         or (entity_1[2][1] >= entity_2[2][0] and entity_1[2][1] <= entity_2[2][1] and len(entity_1[1]) <= len(entity_2[1]))):
                return True

    return False

def delete_entity_item(temp_str_list, entity_1, list_relation_len):
    for another_index, item in enumerate(temp_str_list[:list_relation_len]):
        entity_pair_comp = (item[0], item[1])
        for entity_2 in entity_pair_comp:
            if entity_1 != entity_2 and ((entity_1[2][0] >= entity_2[2][0] and entity_1[2][1] <= entity_2[2][1])
                                         or (entity_1[2][0] <= entity_2[2][0] and entity_1[2][1] >= entity_2[2][1])
                                         or (entity_1[2][0] >= entity_2[2][0] and entity_1[2][0] <= entity_2[2][1])
                                         or (entity_1[2][1] >= entity_2[2][0] and entity_1[2][1] <= entity_2[2][1])):
                return True
    for another_index, item in enumerate(temp_str_list[list_relation_len:]):
        entity_2 = item[0]
        if entity_1 != entity_2 and ((entity_1[2][0] >= entity_2[2][0] and entity_1[2][1] <= entity_2[2][1])
                                     or (entity_1[2][0] >= entity_2[2][0] and entity_1[2][0] <= entity_2[2][1] and len(entity_1[1]) <= len(entity_2[1]))
                                     or (entity_1[2][1] >= entity_2[2][0] and entity_1[2][1] <= entity_2[2][1] and len(entity_1[1]) <= len(entity_2[1]))):
            return True
    return False

def make_csv(xml, f, type1, type2):
    tree = etree.parse(xml)
    document = tree.getroot()
    for sentence in document:
        iid = sentence.attrib['id'].split('.')
        sentence_id = iid[-2][1:] + iid[-1][1:]
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
                entity_dict[entity_id] = (entity_id, entity_text, (int(entity_start), int(entity_end) + 1), entity_type)
                entity_non_relation[entity_id] = [(entity_id, entity_text, (int(entity_start), int(entity_end) + 1), entity_type)]
        pair_list = []
        for entity_or_pair in sentence:
            if entity_or_pair.tag == 'pair':
                if entity_or_pair.attrib['ddi'] == 'true':
                    pair = entity_or_pair
                    pair_id = pair.attrib['id']
                    pair_type = pair.attrib['type']
                    pair_subj_id = pair.attrib['e1']
                    pair_obj_id = pair.attrib['e2']
                    if pair_type not in type2:
                        type2.append(pair_type)
                    if pair_subj_id in entity_non_relation.keys():
                        entity_non_relation.pop(pair_subj_id)
                    if pair_obj_id in entity_non_relation.keys():
                        entity_non_relation.pop(pair_obj_id)

                    pair_list.append([entity_dict[pair_subj_id], entity_dict[pair_obj_id], pair_type, pair_id])
        anntotaions = pair_list
        list_relation_len = len(pair_list)
        anntotaions.extend(list(entity_non_relation.values()))

        delete_item_list = []
        delete_item_dict = {}
        for index, relation_item in enumerate(anntotaions[:list_relation_len]):
            delete_flag_entity_1 = delete_relation_item(anntotaions, relation_item[0], list_relation_len)
            delete_flag_entity_2 = delete_relation_item(anntotaions, relation_item[1], list_relation_len)
            if delete_flag_entity_1 and delete_flag_entity_2:
                delete_item_dict[str(relation_item)] = 2
            elif delete_flag_entity_1:
                delete_item_dict[str(relation_item)] = 1
            elif delete_flag_entity_2:
                delete_item_dict[str(relation_item)] = 0
        for index, entity_1 in enumerate(anntotaions[list_relation_len:]):
            delete_flag = delete_entity_item(anntotaions, entity_1[0], list_relation_len)
            if delete_flag:
                delete_item_list.append(entity_1)

        print(delete_item_dict, delete_item_list)
        for item, num in delete_item_dict.items():
            item = eval(item)
            anntotaions.remove(item)
            if num != 2:
                anntotaions.append([item[num]])
        for item in delete_item_list:
            anntotaions.remove(item)

        if anntotaions != []:
            f.write(f'{sentence_id}||{sentence_text}||{str(anntotaions)}\n')
    return type1, type2

if __name__ == '__main__':
    raw_data_dir_list = ['../raw_data/Train/DrugBank', '../raw_data/Train/MedLine',
                         '../raw_data/Test/DrugBank', '../raw_data/Test/MedLine']
    res_dir = '../data_csv'
    entity_type = []
    relation_type = []
    for raw_data_dir in raw_data_dir_list:
        with open(os.path.join('../data_csv', raw_data_dir.split('/')[-2] + '_' + raw_data_dir.split('/')[-1]+'.csv'), 'w', encoding='utf-8') as f:
            for _, _, xml_list in os.walk(raw_data_dir):
                for xml in xml_list:
                    raw_data_xml = os.path.join(raw_data_dir, xml)
                    entity_type, relation_type = make_csv(raw_data_xml, f, entity_type, relation_type)
    print('entity_type =', entity_type)
    print('relation_type =', relation_type)

