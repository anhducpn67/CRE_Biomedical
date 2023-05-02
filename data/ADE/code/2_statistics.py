def statistics_corpus_relation(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()

    print("* Relation statistics")
    relation_list = ["Drug_Disease_interaction"]
    num_relation_dic = {}
    for relation in relation_list:
        num_relation_dic.setdefault(relation, 0)

    total_relation_num = 0
    for i in data:
        data_dic = eval(i)
        for k, v in data_dic.items():
            if k in relation_list:
                num_relation_dic[k] = num_relation_dic[k] + len(v)
                total_relation_num += len(v)

    for k, v in num_relation_dic.items():
        print(k, v, sep=": ")
    print("total relation num:", total_relation_num)


def statistics_corpus_entity(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    print(f"* Sent. count: {len(data)}")

    print("* Entity statistics")
    entity_list = ["Drug", "Disease"]
    num_entity_dict = {}
    for entity in entity_list:
        num_entity_dict.setdefault(entity, 0)

    total_entity_num = 0
    for i in data:
        data_dic = eval(i)
        for k, v in data_dic.items():
            if k in num_entity_dict:
                num_entity_dict[k] = num_entity_dict[k] + len(v)
                total_entity_num += len(v)

    for k, v in num_entity_dict.items():
        print(k, v, sep=": ")
    print("total entity num:", total_entity_num)


train_file = "../BIOES/base/ADE_train_base_model_data.json"
valid_file = "../BIOES/base/ADE_valid_base_model_data.json"
test_file = "../BIOES/base/ADE_test_base_model_data.json"
file_list = [train_file, valid_file, test_file]
for file in file_list:
    print("*****", file)
    statistics_corpus_entity(file)
    statistics_corpus_relation(file)
    print("="*50)
