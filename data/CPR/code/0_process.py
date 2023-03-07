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


def process_data(abstracts_file, entities_file, relations_file, gold_file, res_file):
    with open(abstracts_file, "r") as f:
        abstracts_data = f.readlines()

    with open(entities_file, "r") as f:
        entities_data = f.readlines()

    with open(relations_file, "r") as f:
        relations_data = f.readlines()

    with open(gold_file, "r") as f:
        gold_data = f.readlines()

    dic_PMID_abstract = {}
    for i in abstracts_data:
        temp = i.split("\t")
        PMID = temp[0]
        dic_PMID_abstract[PMID] = (temp[1] + "\t" + temp[2]).replace("\n", "")
        # dic_PMID_abstract[PMID] = temp[1] + "\t" + temp[2]

    dic_PMID_entities_temp = {}
    for i in entities_data:
        temp = i.split("\t")
        PMID = temp[0]
        dic_PMID_entities_temp.setdefault(PMID, [])

        term_number = temp[1]
        type_of_entity_mention = temp[2]
        start_span = int(temp[3])
        end_span = int(temp[4])
        string_entity = temp[5]
        dic_PMID_entities_temp[PMID].append((term_number, type_of_entity_mention, (start_span, end_span), string_entity))

    dic_PMID_entities = {}
    for PMID, tags_list in dic_PMID_entities_temp.items():
        dic_entityNumber_string = {}
        dic_entityNumber_span = {}
        dic_entityNumber_type = {}

        for tag_tuple in tags_list:
            term_number, type_of_entity_mention, start_end_span, string_entity = tag_tuple
            if (term_number in dic_entityNumber_string.keys()) or (term_number in dic_entityNumber_span.keys()) or \
                    (term_number in dic_entityNumber_type.keys()):
                raise Exception("entity error!")

            dic_entityNumber_string[term_number] = string_entity.replace("\n", "")
            dic_entityNumber_span[term_number] = start_end_span
            dic_entityNumber_type[term_number] = type_of_entity_mention

        dic_PMID_entities[PMID] = [dic_entityNumber_string, dic_entityNumber_span, dic_entityNumber_type]


    dic_PMID_relations_temp = {}
    for i in relations_data:
        temp = i.split("\t")
        PMID = temp[0]
        dic_PMID_relations_temp.setdefault(PMID, [])

        relation_group = temp[1].replace("\n", "")
        if_evl = temp[2].replace("\n", "")
        relation = temp[3].replace("\n", "")
        entity_1 = temp[4].replace("\n", "")
        entity_2 = temp[5].replace("\n", "")
        dic_PMID_relations_temp[PMID].append((relation_group, if_evl, relation, entity_1, entity_2))

    dic_PMID_relations = {}
    for PMID, tags in dic_PMID_relations_temp.items():
        dic_temp = {}
        for tag_tuple in tags:
            relation_group, if_evl, relation, entity_1, entity_2 = tag_tuple

            entity_1 = entity_1.split(":")[1]
            entity_2 = entity_2.split(":")[1]

            key_entity = (entity_1, entity_2)
            assert int(entity_1.replace("T", "")) < int(entity_2.replace("T", ""))
            dic_temp.setdefault(key_entity, [])
            dic_temp[key_entity].append([relation_group, if_evl.strip(), relation])
        dic_PMID_relations[PMID] = dic_temp

    dic_PMID_gold = {}
    for i in gold_data:
        temp = i.split("\t")
        PMID = temp[0]

        relation_group = temp[1]
        entity_1 = temp[2]
        entity_2 = temp[3]

        dic_PMID_gold[PMID] = [relation_group, entity_1, entity_2]

    count = 0
    delete_item_list_count = 0
    delete_item_dict_count = 0

    total_entity_count = 0
    total_relation_count = 0
    res_list = []
    for PMID in dic_PMID_abstract.keys():
        process_str = ""
        process_str = process_str + str(PMID) + "||"
        process_str = process_str + str(dic_PMID_abstract[PMID]) + "||"

        temp_str_list = []
        if PMID in dic_PMID_relations.keys():
            """dic_PMID_relations[PMID] = dic_entitesPair_tags[(str(entity_1), str(entity_2))] = [relation_group, if_evl, relation] """
            entity_in_relation_list = []
            for entity_pair, tags_list in dic_PMID_relations[PMID].items():
                entity_1, entity_2 = entity_pair
                entity_in_relation_list.append(entity_1)
                entity_in_relation_list.append(entity_2)

                for tag_tuple in tags_list:
                    relation_group, if_evl, relation = tag_tuple
                    add_str = [(entity_1, dic_PMID_entities[PMID][0][entity_1], dic_PMID_entities[PMID][1][entity_1], dic_PMID_entities[PMID][2][entity_1]),
                               (entity_2, dic_PMID_entities[PMID][0][entity_2], dic_PMID_entities[PMID][1][entity_2], dic_PMID_entities[PMID][2][entity_2]),
                               relation_group, if_evl, relation]
                    temp_str_list.append(add_str)
        else:
            entity_in_relation_list = []

        list_relation_len = len(temp_str_list)
        total_relation_count +=list_relation_len
        if PMID in dic_PMID_entities.keys():
            """dic_PMID_entities[PMID] = [dic_entityNumber_string, dic_entityNumber_span, dic_entityNumber_type]"""
            dic_entityNumber_string, dic_entityNumber_span, dic_entityNumber_type = dic_PMID_entities[PMID]
            for entity_term_ID in dic_entityNumber_string.keys():
                if entity_term_ID not in entity_in_relation_list:
                    add_str = [(entity_term_ID, dic_entityNumber_string[entity_term_ID], dic_entityNumber_span[entity_term_ID], dic_entityNumber_type[entity_term_ID])]
                    temp_str_list.append(add_str)

        total_entity_count += len(temp_str_list) + list_relation_len
        delete_item_list = []
        delete_item_dict = {}
        for index, relation_item in enumerate(temp_str_list[:list_relation_len]):
            delete_flag_entity_1 = delete_relation_item(temp_str_list, relation_item[0], list_relation_len)
            delete_flag_entity_2 = delete_relation_item(temp_str_list, relation_item[1], list_relation_len)
            if delete_flag_entity_1 and delete_flag_entity_2:
                delete_item_dict[str(relation_item)] = 2
            elif delete_flag_entity_1:
                delete_item_dict[str(relation_item)] = 1
            elif delete_flag_entity_2:
                delete_item_dict[str(relation_item)] = 0
        for index, entity_1 in enumerate(temp_str_list[list_relation_len:]):
            delete_flag = delete_entity_item(temp_str_list, entity_1[0], list_relation_len)
            if delete_flag:
                delete_item_list.append(entity_1)
        delete_item_list_count += len(delete_item_list)
        delete_item_dict_count += len(delete_item_dict)*2

        # print('PMID', PMID, delete_item_list)
        # print('temp_str_list_old', temp_str_list)
        for item, num in delete_item_dict.items():
            item = eval(item)
            temp_str_list.remove(item)
            if num != 2:
                temp_str_list.append([item[num]])
                count+=1
        for item in delete_item_list:
            temp_str_list.remove(item)

        process_str = process_str + str(temp_str_list)
        res_list.append(process_str)
    delete_entity = delete_item_list_count + delete_item_dict_count - count
    print("total_relation", total_relation_count)
    print("total_entity", total_entity_count)
    print("delete_entity", delete_entity)
    print("delete_relation", delete_item_dict_count/2)
    print("+++================++++")
    print("+++================++++")

    with open(res_file, "w") as f:
        for i in res_list:
            f.write(i+"\n")


def process_train_data():
    train_abstracts_file = "../raw_data/chemprot_training/chemprot_training_abstracts.tsv"
    train_entities_file = "../raw_data/chemprot_training/chemprot_training_entities.tsv"
    train_relations_file = "../raw_data/chemprot_training/chemprot_training_relations.tsv"
    train_gold_file = "../raw_data/chemprot_training/chemprot_training_gold_standard.tsv"
    res_file = "../data_csv/CPR_train.csv"
    process_data(train_abstracts_file, train_entities_file, train_relations_file, train_gold_file, res_file)


def process_valid_data():
    valid_abstracts_file = "../raw_data/chemprot_development/chemprot_development_abstracts.tsv"
    valid_entities_file = "../raw_data/chemprot_development/chemprot_development_entities.tsv"
    valid_relations_file = "../raw_data/chemprot_development/chemprot_development_relations.tsv"
    valid_gold_file = "../raw_data/chemprot_development/chemprot_development_gold_standard.tsv"
    res_file = "../data_csv/CPR_valid.csv"
    process_data(valid_abstracts_file, valid_entities_file, valid_relations_file, valid_gold_file, res_file)


def process_test_data():
    test_abstracts_file = "../raw_data/chemprot_test_gs/chemprot_test_abstracts_gs.tsv"
    test_entities_file = "../raw_data/chemprot_test_gs/chemprot_test_entities_gs.tsv"
    test_relations_file = "../raw_data/chemprot_test_gs/chemprot_test_relations_gs.tsv"
    test_gold_file = "../raw_data/chemprot_test_gs/chemprot_test_gold_standard.tsv"
    res_file = "../data_csv/CPR_test.csv"
    process_data(test_abstracts_file, test_entities_file, test_relations_file, test_gold_file, res_file)


process_train_data()
process_valid_data()
process_test_data()















