import spacy
import pandas as pd
import re
import nltk
import collections
from tqdm import tqdm
from transformers import *
import unicodedata
import json
import os
from itertools import combinations

global error
error = 0

def re_search_token(token):
    token = re.sub("\(", "\\(", token)
    token = re.sub("\)", "\\)", token)
    token = re.sub("\?", "\\?", token)
    token = re.sub("\+", "\\+", token)
    token = re.sub("\$", "\\$", token)
    token = re.sub("\[", "\\[", token)
    token = re.sub("\]", "\\]", token)
    token = re.sub("\.", "\\.", token)
    return token

def get_triple_O_seg(targets_sentence):
    """ this Function ask the last tag must be "O"
        input : ['O', 'O', 'O', 'O', 'B', 'I'...]
    """
    total_list = []
    gold_list = []
    for index in range(len(targets_sentence)):
        # if targets_sentence[index] == "O" or targets_sentence[index][-1] =="E":
        if targets_sentence[index] == "O" or targets_sentence[index] == '[PAD]':
            if len(gold_list) > 0:
                total_list.append(gold_list)
                gold_list = []
        else:
            # if gold_list[-1][1].split("_")[-1]=="E":
            temp_list = targets_sentence[index].split("_")
            if len(temp_list) == 2:
                add_str = temp_list[0]+"_"+temp_list[1]
            else:
                add_str = targets_sentence[index]

            gold_list.append((index, add_str))

            if add_str[-1]=="E" or add_str[-1]=="S":
                total_list.append(gold_list)
                gold_list = []

            if index == len(targets_sentence)-1:
                if len(gold_list)>0:
                    total_list.append(gold_list)

    return total_list

def unicode_to_ascii(s):
    temp_list = []
    for c in unicodedata.normalize('NFD', s):
        if unicodedata.category(c) != 'Mn':
            if unicodedata.category(c) == 'Co':
                c = " "
                # print("replace Co once !")
            temp_list.append(c)
        else:
            pass
            # print("delete Mn Once !")
    return "".join(temp_list)


def generated_tagged_index(token_list, entity_span, sentence):
    """ Convert sent-level annotation to bert-token-level annotation for each entity.
       entity = [('T6', 'acetylcholinesterase', (35, 55), 'GENE-Y')]
   """
    index_list = []
    sentence_making = ""
    pos_start = int(entity_span[0])
    pos_end = int(entity_span[1])

    sentence = sentence.replace("\t", " ").replace(" ", " ").replace(" ", " ").replace(" ", " ").replace(" ", " ").replace(" ", " ")
    for i in range(len(token_list)):
        start_counter = len(sentence_making)
        new_token = token_list[i].replace("##", "").lower()

        if new_token == '[unk]':
            sent_left = sentence.replace(sentence_making,"")
            next_token = re_search_token(token_list[i+1])
            unk_span = re.search(next_token, sent_left).span()
            unk_token = sentence.replace(sentence_making,"")[:unk_span[0]]
            new_token = unk_token
            # print("unk replaced once !")
        if sentence_making + new_token + "   " == sentence[: len(sentence_making) + len(new_token) + 3]:
            sentence_making = sentence_making + new_token + "  "
            end_counter = len(sentence_making) - 4
        elif sentence_making + new_token + "  " == sentence[: len(sentence_making) + len(new_token) + 2]:
            sentence_making = sentence_making + new_token + "  "
            end_counter = len(sentence_making) - 3
        elif sentence_making + new_token + " " == sentence[: len(sentence_making)+len(new_token)+1]:
            sentence_making = sentence_making + new_token + " "
            end_counter = len(sentence_making)-2
        elif sentence_making + new_token == sentence[: len(sentence_making) + len(new_token)]:
            sentence_making = sentence_making + new_token
            end_counter = len(sentence_making)-1

        else:
            raise NameError("sentence making error!")

        if start_counter >= int(pos_start) and end_counter <= int(pos_end) and (new_token in sentence[pos_start: pos_end]):
            index_list.append(i)
    assert sentence_making == sentence

    return index_list


def generated_BIOES(tagged_index_list, temp_list):

    if len(tagged_index_list) == 0:
        temp_list = temp_list
    if len(tagged_index_list) == 1:
        temp_list[tagged_index_list[0]] ="S"
    if len(tagged_index_list) == 2:
        temp_list[tagged_index_list[0]] =  "B"
        temp_list[tagged_index_list[1]] = "E"
    if len(tagged_index_list) > 2:
        temp_list[tagged_index_list[0]] = "B"
        for i in tagged_index_list[1:-1]:
            temp_list[i] =  "I"
        temp_list[tagged_index_list[-1]] =  "E"
    return temp_list


def process_annotation_Entity_Span(annotation, sent, token_list):
    """
    anno_item = [('T1', 'edrophonium', (641, 652), 'CHEMICAL'), ('T5', 'Torpedo californica AChE', (579, 603), 'GENE-Y'), 'CPR:4', 'Y', 'INHIBITOR']
              / [('T28', 'epidermal growth factor receptor', (1081, 1113), 'GENE-Y')]
    """

    sent = sent.lower()
    temp_list = ["O"] * len(token_list)
    exist_one_whole_list = []
    exist_each_entity_list = []
    delete_list = []
    delete_entity_list = []

    for anno_item in annotation:
        if len(anno_item) == 5:
            entity_list = anno_item[0:2]
        elif len(anno_item) == 1:
            entity_list = anno_item
        else:
            raise Exception("anno_item error !")

        for entity in entity_list:
            # entity_type = entity[-1]
            entity_span = entity[2]
            tagged_index_list = generated_tagged_index(token_list, entity_span, sent)
            generated_tagged_index_list = [i+1 for i in tagged_index_list]

            if len(generated_tagged_index_list)>0:
                if  generated_tagged_index_list[-1]<=510 :
                    """ 
                    exist nested entity, if first token of entity already in the list, skip the second token
                    if when we skip one entity, delete related anno_item 
                    """
                    if str(generated_tagged_index_list) not in exist_each_entity_list:
                        if not list(set(generated_tagged_index_list) & set(exist_one_whole_list)):
                            # if  (generated_tagged_index_list[0] not in exist_one_whole_list) and (generated_tagged_index_list[-1] not in exist_one_whole_list)  :
                            exist_one_whole_list.extend(generated_tagged_index_list)
                            exist_each_entity_list.append(str(generated_tagged_index_list))
                            temp_list = generated_BIOES(tagged_index_list, temp_list)
                        else:
                            delete_list.append(str(anno_item))
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            for entity in entity_list:
                                entity_span = entity[2]
                                tagged_index_list = generated_tagged_index(token_list, entity_span, sent)
                                generated_tagged_index_list = [i+1 for i in tagged_index_list]
                                delete_entity_list.append(str(generated_tagged_index_list))

                else:
                    # print("entity span out of 510 !")
                    delete_list.append(str(anno_item))
                    delete_entity_list.append(str(generated_tagged_index_list))

    if len(temp_list)>510:
        temp_list = temp_list[:510]
    temp_list.insert(0, "O")
    temp_list.append("O")
    # print(delete_list)
    delete_list_dict = collections.Counter(delete_list)
    for anno, num in delete_list_dict.items():
        anno = eval(anno)
        annotation.remove(anno)
        if len(anno) == 5 and num == 1:
            if anno[0][2][1] < anno[1][2][1]:
                annotation.append([anno[0]])
            else:
                annotation.append([anno[1]])

    for entity in delete_entity_list:
        try:
            exist_each_entity_list.remove(entity)
        except:
            pass

    return temp_list, annotation, exist_each_entity_list


def process_annotation_Entity_type(annotation, sent, token_list, entity_type_flag):
    """ sent level """
    sent = sent.lower()
    joint_gold_list = ["O"] * len(token_list)
    only_entity_type = []
    exist_one_whole_list = []

    """anno_item = [('T1', 'edrophonium', (641, 652), 'CHEMICAL'), ('T5', 'Torpedo californica AChE', (579, 603), 'GENE-Y'), 'CPR:4', 'Y', 'INHIBITOR']"""
    for anno_item in annotation:
        if len(anno_item) == 5:
            entity_list = anno_item[0:2]
        elif len(anno_item) == 1:
            entity_list = anno_item
        else:
            print(anno_item)
            raise Exception("anno_item error !")

        for entity in entity_list:
            entity_type = entity[-1]

            if entity_type=="CHEMICAL":
                entity_type = "Drug"
            elif entity_type=="GENE-Y" or entity_type=="GENE-N":
                entity_type = "Gene"

            # if entity_type.replace("-", "_") == entity_type_flag:
            if entity_type == entity_type_flag:
                entity_span = entity[2]
                tagged_index_list = generated_tagged_index(token_list, entity_span, sent)

                """exist nested entity, if first entity already in the list, skip the second """
                """so we need put the relation pair in the first to extract relation triple in priority """
                if len(tagged_index_list)>0 :
                    if tagged_index_list[-1]<=510:
                        #deal only_entity_type cls and sep
                        generated_entity_span =[i+1 for i in tagged_index_list]

                        if str(generated_entity_span) not in only_entity_type:
                            if not list(set(generated_entity_span) & set(exist_one_whole_list)):
                                exist_one_whole_list.extend(generated_entity_span)
                                only_entity_type.append(str(generated_entity_span))
                                joint_gold_list = generated_BIOES(tagged_index_list, joint_gold_list)
                            else:
                                raise Exception("there is a nested entity not delete in entity span")
                    else:
                        raise Exception("there is en entity out of range 512, which not deleted in entity span ")

    #have deal only_entity_type above
    if len(joint_gold_list)>510:
        joint_gold_list = joint_gold_list[:510]
    joint_gold_list.insert(0, "O")
    joint_gold_list.append("O")

    return joint_gold_list, only_entity_type


def process_annotation_Relation_type(annotation, sent, token_list, relation, entity_list):
    sent = sent.lower()
    relation_entity_pair_list = []
    existing_list = []
    global error_count
    error_count = 0
    """anno_item = [('T1', 'edrophonium', (641, 652), 'CHEMICAL'), ('T5', 'Torpedo californica AChE', (579, 603), 'GENE-Y'), 'CPR:4', 'Y', 'INHIBITOR']"""
    for anno_item in annotation:
        if len(anno_item) == 5:
            relation_tag = anno_item[2]

            # if relation_tag in ["CPR:1","CPR:2","CPR:3","CPR:4","CPR:5","CPR:6","CPR:7","CPR:8","CPR:9"]:
            if relation_tag in ["CPR:1","CPR:2","CPR:3","CPR:4","CPR:5","CPR:6","CPR:7","CPR:8","CPR:9"]:
                relation_tag = "Drug_Gene_interaction"
            else:
                relation_tag = "Wrong"

            if relation_tag == relation:
                subject_token_span = generated_tagged_index(token_list, anno_item[0][2], sent)
                object_token_span = generated_tagged_index(token_list, anno_item[1][2], sent)
                subject_token_span = [i+1 for i in subject_token_span]
                object_token_span = [i+1 for i in object_token_span]
                try:
                    assert len(subject_token_span)>0
                    assert len(object_token_span)>0
                    existing_list.append(subject_token_span)
                    existing_list.append(object_token_span)

                    if (str(subject_token_span) in entity_list) and (str(object_token_span) in entity_list):
                        relation_entity_pair_list.append(str((subject_token_span, object_token_span)))
                    else:
                        print("Relation_extraction nested entity..............", subject_token_span, object_token_span)
                except AssertionError:
                    error_count +=1
                    print("error in paring relation! ", error_count)

    return relation_entity_pair_list


def return_sentence_pos(sentences, text):
    #return start and end position for each sentence
    sentence_start_end_list = []

    for sentence in sentences:
        # try:
        sentence = re.sub("\(", "\\(", sentence)
        sentence = re.sub("\)", "\\)", sentence)
        sentence = re.sub("\?", "\\?", sentence)
        sentence = re.sub("\+", "\\+", sentence)
        sentence = re.sub("\$", "\\$", sentence)
        sentence = re.sub("\[", "\\[", sentence)
        sentence = re.sub("\]", "\\]", sentence)
        sentence = re.sub("\*", "\\*", sentence)
        sentence = re.sub("\.", "\\.", sentence)

        temp_finditer = list(re.finditer(sentence, text))
        if len(temp_finditer) == 0:
            raise NameError("no find sentence in text")

        temp_count = 0
        for m in temp_finditer:
            if len(temp_finditer) == 1:
                sentence_start_end_list.append((m.start(), m.end()))
            if len(temp_finditer) > 1 and temp_count <= len(temp_finditer):
                sentence_start_end_list.append(temp_finditer[temp_count].span())
                temp_count += 1
                break

        # except:
        #     print("re.finder error~~~~~~~~~~~~~~~~~~~~~~")

    return sentence_start_end_list

def convert_new_anno(doc_annotation, sent_span):
    B_sent_span = sent_span[0]
    new_annotation = []
    for anno in doc_annotation:
        temp_anno =[]
        if len(anno)==5:
            new_entity_1_S = anno[0][2][0] - B_sent_span
            new_entity_1_E = anno[0][2][1] - B_sent_span
            new_entity_1_span = (new_entity_1_S, new_entity_1_E)
            new_entity_1 = (anno[0][0], anno[0][1], new_entity_1_span, anno[0][3])

            new_entity_2_S = anno[1][2][0] - B_sent_span
            new_entity_2_E = anno[1][2][1] - B_sent_span
            new_entity_2_span = (new_entity_2_S, new_entity_2_E)
            new_entity_2 =  (anno[1][0], anno[1][1], new_entity_2_span, anno[1][3])

            if new_entity_1_S < new_entity_2_S:
                temp_anno.append(new_entity_1)
                temp_anno.append(new_entity_2)
            else:
                temp_anno.append(new_entity_2)
                temp_anno.append(new_entity_1)

            temp_anno.append(anno[2])
            temp_anno.append(anno[3])
            temp_anno.append(anno[4])

        elif len(anno)==1:
            new_entity_1_S = anno[0][2][0] - B_sent_span
            new_entity_1_E = anno[0][2][1] - B_sent_span
            new_entity_1_span = (new_entity_1_S, new_entity_1_E)
            new_entity_1 = (anno[0][0], anno[0][1], new_entity_1_span, anno[0][3])
            temp_anno.append(new_entity_1)
        else:
            raise Exception("anno wrong !")

        new_annotation.append(temp_anno)
    return new_annotation

def sep_anno(new_sent_span_list, doc_annotation):

    doc_anno_list_sep_list = [[] for i in range(len(new_sent_span_list))]
    count = 0
    for anno in eval(doc_annotation):
        if len(anno)==5:
            min_span = min(anno[0][2][0], anno[1][2][0])
            max_span = max(anno[0][2][1], anno[1][2][1])
        elif len(anno)==1:
            min_span = anno[0][2][0]
            max_span = anno[0][2][1]
        else:
            raise Exception("anno wrong !")

        for sent_index, sent_span in enumerate(new_sent_span_list):
            if min_span >= sent_span[0] and max_span <= sent_span[1]:
                doc_anno_list_sep_list[sent_index].append(anno)
                count +=1
                break

    del_num =  len(eval(doc_annotation)) - count
    return doc_anno_list_sep_list, del_num


def process_data(raw_file, tokenizer, res_file, entity_type_list, relation_type_list):
    if os.path.exists(res_file):
        os.remove(res_file)
        print(res_file+" has deleted!!")

    with open(raw_file, "r") as f:
        raw_data = f.readlines()

    totol_del_num = 0
    write_data_list = []
    with tqdm(total=len(raw_data)) as pbar:
        for sent_index in range(len(raw_data)):
            pbar.update(1)
            temp_res = raw_data[sent_index].split("||")
            assert len(temp_res)==3
            doc_ID, doc, doc_annotation = temp_res
            doc = unicode_to_ascii(doc)
            sentences_list = nltk.sent_tokenize(doc)
            new_sent_span_list = return_sentence_pos(sentences_list, doc)
            doc_anno_list_sep_list, del_num = sep_anno(new_sent_span_list, doc_annotation)
            totol_del_num+=del_num

            assert len(sentences_list)==len(new_sent_span_list)
            assert len(sentences_list)==len(doc_anno_list_sep_list)

            for sent_index, sentence in enumerate(sentences_list):
                new_ID = "11111"+str(doc_ID)+str(sent_index)
                new_token_list = tokenizer.tokenize(sentence)
                new_annotation = convert_new_anno(doc_anno_list_sep_list[sent_index], new_sent_span_list[sent_index])
                if len(new_annotation)>0:
                    write_data = precess_BIOES(new_ID, new_token_list, sentence, new_annotation, entity_type_list, relation_type_list)
                    write_data_list.append(write_data)

    print(totol_del_num)

    if raw_file =='../data_csv/CPR_valid.csv' :
        new_len=len(write_data_list)
        # write_data_list = sorted(write_data_list, key=lambda x: len(x["relation_Drug_Gene_interaction"]), reverse=True)[:int(new_len/5)]
        write_data_list = write_data_list[:int(new_len/5)]
    if raw_file =='../data_csv/CPR_test.csv':
        new_len=len(write_data_list)
        # write_data_list = sorted(write_data_list, key=lambda x: len(x["relation_Drug_Gene_interaction"]), reverse=True)[:int(new_len/5)]
        write_data_list = write_data_list[:int(new_len/6)]

    with open(res_file, "w") as f:
        for i in write_data_list:
            f.write(json.dumps(i))
            f.write("\n")

def precess_BIOES(ID, token_list, sent, annotation, entity_type_list, relation_type_list):

    token_index_list2 = [tokenizer.convert_tokens_to_ids(i) for i in token_list]
    if len(token_index_list2)>510:
        token_index_list2 = token_index_list2[:510]
    token_index_list2.insert(0, tokenizer.convert_tokens_to_ids("[CLS]"))
    token_index_list2.append(tokenizer.convert_tokens_to_ids("[SEP]"))

    tag_list_BIOES, new_annotation, entity_span_entity_list = process_annotation_Entity_Span(annotation, sent, token_list)
    raw_sep_entity = get_triple_O_seg(tag_list_BIOES)
    sep_entity = []
    for i in raw_sep_entity:
        temp_list = []
        for j in i:
            temp_list.append(j[0])
        sep_entity.append(str(temp_list))

    tag_list_only_entity_type = {}
    tag_list_joint_entity_type = {}
    for entity_type in entity_type_list:

        joint_gold_list, only_entity_type = process_annotation_Entity_type(new_annotation, sent, token_list, entity_type)
        tag_list_only_entity_type["only_entity_type_"+entity_type] = only_entity_type
        tag_list_joint_entity_type["joint_entity_type_"+entity_type] = joint_gold_list
        for entity_tag_index, token_tag in enumerate(joint_gold_list):
            if token_tag!="O":
                assert token_tag == tag_list_BIOES[entity_tag_index]
        for entity in only_entity_type:
            for j in eval(entity):
                assert joint_gold_list[j] == tag_list_BIOES[j]

    tag_list_joint_entity_type_combined = ["O"]*len(tag_list_BIOES)
    for key, v in tag_list_joint_entity_type.items():
        for index, tag in enumerate(v):
            if tag != "O":
                assert tag_list_joint_entity_type_combined[index] == "O"
                tag_list_joint_entity_type_combined[index] = key.split("_")[-1]+"_"+tag

    entity_type_entity_list = []
    for entity_type in tag_list_only_entity_type.values():
        entity_type_entity_list.extend(entity_type)
    try:
        assert sorted(entity_span_entity_list) == sorted(entity_type_entity_list)
    except:
        wrong_entity = set(entity_span_entity_list).difference(set(entity_type_entity_list))
        print(wrong_entity)
        pass

    tag_list_Relation = {}
    sampled_entity_span = []
    for relation in relation_type_list:
        relation_entity_pair_list = process_annotation_Relation_type(new_annotation, sent, token_list, relation, entity_type_entity_list)
        relation_entity_pair_list = [str(tuple(sorted(eval(i)))) for i in relation_entity_pair_list]
        tag_list_Relation["relation_"+relation.replace(":", "_")] = list(set(relation_entity_pair_list))
        for i in relation_entity_pair_list:
            sampled_entity_span.append(str(eval(i)[0]))
            sampled_entity_span.append(str(eval(i)[1]))
    sampled_entity_span = list(set(sampled_entity_span))

    all_combined_list = sorted(list(combinations(sep_entity, 2)))
    has_relation_list = []
    for each_relation_list in list(tag_list_Relation.values()):
        if each_relation_list:
            for pair in each_relation_list:
                has_relation_list.append(tuple([str(i) for i in sorted(eval(pair))]))
    None_relation = []
    for x in all_combined_list:
        if x not in has_relation_list:
            None_relation.append(str(tuple([eval(i) for i in x])))
    # None_relation = [str(tuple([eval(i) for i in x])) for x in all_combined_list if x not in sorted(list(tag_list_Relation.values()))]
    tag_list_Relation["relation_None"] = None_relation

    res_str_dic = {}
    res_str_dic["ID"] = ID
    res_str_dic["tokens"] = token_index_list2
    res_str_dic["entity_span"] = tag_list_BIOES
    res_str_dic["sep_entity"] = sep_entity
    # res_str_dic["sampled_entity_span"] = sampled_entity_span
    res_str_dic["sampled_entity_span"] = sep_entity
    res_str_dic["entity_span_and_type"] = tag_list_joint_entity_type_combined

    res_str_dic.update(tag_list_only_entity_type)
    res_str_dic.update(tag_list_joint_entity_type)
    res_str_dic.update(tag_list_Relation)

    assert len(token_index_list2) == len(tag_list_BIOES)
    assert len(tag_list_BIOES) == len(joint_gold_list)
    assert len(tag_list_BIOES) <= 512

    return dict(sorted(res_str_dic.items(), key=lambda item:len(item[0])))



if __name__ == '__main__':
    # clear multi
    multi_file = "../../Multi_Task_Training/base"
    for root,dirs,files in os.walk(multi_file):
        for file in files:
            os.remove(os.path.join(root, file))
    multi_file_test = "../../Multi_Task_Training/base/test"
    for root,dirs,files in os.walk(multi_file_test):
        for file in files:
            os.remove(os.path.join(root, file))


    # bert_model = "base"
    bert_model = "base"

    if bert_model == "base":
        model_path = "../../../../../Data/embedding/biobert_base"
        tokenizer = BertTokenizer.from_pretrained(model_path)
    elif bert_model == "large":
        model_path = "../../../../../Data/embedding/biobert_large"
        tokenizer = BertTokenizer.from_pretrained(model_path)


    dir_raw = "../data_csv"
    dir_res = "../BIOES/" + bert_model

    file_list = ["CPR_train.csv","CPR_valid.csv","CPR_test.csv"]
    # file_list = ["CPR_test.csv"]
    # entity_type_list = ["CHEMICAL", "GENE_Y", "GENE_N"]
    entity_type_list = ["Drug", "Gene"]
    # train_relation_type_list = ["CPR:1","CPR:2","CPR:3","CPR:4","CPR:5","CPR:6","CPR:7","CPR:8","CPR:9","CPR:10"]
    # valid_relation_type_list = ["CPR:1","CPR:2","CPR:3","CPR:4","CPR:5","CPR:6","CPR:7","CPR:8","CPR:9","CPR:10"]
    # train_relation_type_list = ["CPR:1","CPR:2","CPR:3","CPR:4","CPR:5","CPR:6","CPR:7","CPR:8","CPR:9"]
    # valid_relation_type_list = ["CPR:1","CPR:2","CPR:3","CPR:4","CPR:5","CPR:6","CPR:7","CPR:8","CPR:9"]
    train_relation_type_list = ["Drug_Gene_interaction"]
    valid_relation_type_list = ["Drug_Gene_interaction"]
    # train_relation_type_list = ["CPR:3","CPR:4","CPR:5","CPR:6","CPR:9"]
    # valid_relation_type_list = ["CPR:3","CPR:4","CPR:5","CPR:6","CPR:9"]

    for file in file_list:
        if file == "CPR_train.csv":
            relation_type_list = train_relation_type_list
        else:
            relation_type_list = valid_relation_type_list

        raw_file = os.path.join(dir_raw, file)

        name, post_fix = file.split(".")
        res_file = os.path.join(dir_res, name+"_"+bert_model+"_model_data.json")
        process_data(raw_file, tokenizer, res_file, entity_type_list, relation_type_list)
