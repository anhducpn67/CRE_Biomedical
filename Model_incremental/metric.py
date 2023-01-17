import torch
from collections import Counter
from itertools import groupby
from tensorboardX import SummaryWriter
import numpy as np


def switch_dim(one_batch_res_list, one_batch_gold_list, Relation_Type_list, batch_size):
    """
    raw:
        one_batch_res_list = [batch_size(sent_num), sub_num, relation_num]
        one_batch_gold_list = [relation_num, batch_size(sent_num)]
    new:  [batch_size(sent_num), relation_num, sub_num(contact_ob)]
    """
    new_one_batch_res_list = []
    for sent in one_batch_res_list:
        temp_batch_list = []
        for relation_index in range(len(Relation_Type_list)):
            temp_relation_list = []
            for sub in sent:
                temp_relation_list.append(sub[relation_index])
            temp_batch_list.append(temp_relation_list)
        new_one_batch_res_list.append(temp_batch_list)

    new_one_batch_gold_list = []
    for sent_index in range(batch_size):
        temp_batch_list = []
        for one_relation_sent in one_batch_gold_list:
            temp_batch_list.append(one_relation_sent[sent_index])
        new_one_batch_gold_list.append(temp_batch_list)
    return new_one_batch_res_list, new_one_batch_gold_list

def seg_segment(old_one_sent_gold, vocab, sub_num):
    if len(old_one_sent_gold) >1:
        new_one_sent_gold_list = [list(g) for k, g in groupby(old_one_sent_gold, lambda x:x==vocab.stoi["[seg_tag]"]) if not k]
    else:
        new_one_sent_gold_list = torch.Tensor(old_one_sent_gold).repeat(sub_num,1).tolist()

    return new_one_sent_gold_list

def get_sent_len(token_list, vocab):
    sentence_length = 0
    try:
        for token in token_list:
            if int(token) != vocab["[PAD]"]:
                sentence_length += 1
    except:
        print("!")
        pass
    return sentence_length

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

def formatted_outpus(triple_list_list):
    if len(triple_list_list)==0:
        return []

    new_triple_list_list = []
    for triple_list in triple_list_list:
        add_flag = True
        # add only single tag when it is S
        if len(triple_list) == 1:
            if triple_list[0][1][-1] is "S":
                new_triple_list_list.append(triple_list)

        """if one tag of entity is wrong, delete the whole entity"""
        if len(triple_list) > 1:
            """tag must satisfy BIES"""
            if add_flag:
                for triple_index in range(1, len(triple_list) - 1):
                    if triple_list[triple_index][-1][-1] is not "I":
                        add_flag = False
            if add_flag:
                if (triple_list[0][-1][-1] != "B") or (triple_list[-1][-1][-1] != "E"):
                    add_flag = False
            if add_flag:
                new_triple_list_list.append(triple_list)

    return new_triple_list_list

def improved_result(triple_list_list):
    new_triple_list_list = []
    for index, triple_list in enumerate(triple_list_list):
        new_triple_list = []
        index_list = []
        tag_list = []
        pos_list = []

        if len(triple_list) == 1:
            # two strategies: one is for higher precsion, the other is for highter recall
            if triple_list[0][1] != "S":
                new_triple_list_list.append([(triple_list[0][0], "S")])
            # if triple_list[0][1] == "S":
            #     new_triple_list_list.append(triple_list)

        if len(triple_list) > 1:
            # if tag is wrong, fix it
            for triple_index in range(0, len(triple_list)):
                index_list.append(triple_list[triple_index][0])
                pos_list.append(triple_list[triple_index][1][-1])

            for i in range(len(index_list)):
                new_triple_list.append((index_list[i], pos_list[i]))

            # tag must satisfy BIES
            if add_flag:
                for triple_index in range(1, len(triple_list) - 1):
                    if triple_list[triple_index][-1][-1] is not "I":
                        add_flag = False
            if add_flag:
                if (triple_list[0][-1][-1] != "B") or (triple_list[-1][-1][-1] != "E"):
                    add_flag = False

            new_triple_list_list.append(new_triple_list)

    return new_triple_list_list

def combine_all_class_for_total_PRF(each_TP_FN_FP):
    total_TP = 0
    total_FN = 0
    total_FP = 0
    for sub_task_key, TP_FN_FP_list in each_TP_FN_FP.items():
        total_TP += TP_FN_FP_list[0]
        total_FN += TP_FN_FP_list[1]
        total_FP += TP_FN_FP_list[2]

    micro_P, micro_R, micro_F = return_PRF(total_TP, total_FN, total_FP)

    return micro_P, micro_R, micro_F

def return_PRF(TP, FN, FP):
    P = TP / (TP + FP) if (TP + FP) != 0 else 0
    R = TP / (TP + FN) if (TP + FN) != 0 else 0
    F = 2 * P * R / (P + R) if (P + R) != 0 else 0
    return P*100, R*100, F*100

def add_new_relation(total_return_kin_ship, new_dic):
    for key, value in new_dic.items():
        if key in total_return_kin_ship.keys():
            total_return_kin_ship[key] = (total_return_kin_ship[key][0] + value[0],
                                          total_return_kin_ship[key][1] + value[1],
                                          total_return_kin_ship[key][2] + value[2])
        else:
            total_return_kin_ship[key] = value
    return total_return_kin_ship

def accumulated_each_class_TP_FN_FP(pred, tags, task_key, each_entity_TP_FN_FP):
    """ task_key = both sub_task( drug, chemical, ... ) and task(span,type,relation) is ok
    :param pred: segmengted span_List(output of def get_triple_O_seg)
    :param tags: segmengted span_List(output of def get_triple_O_seg)
    :param task_key:
    :return:
    """
    # {'relation_Drug_Disease_interaction': (1194, 35, -1)}
    TP = 0
    new_dic_performance = {}
    # pred = [sorted(eval(i)) for i in pred]
    # tags = [sorted(eval(i)) for i in tags]
    for one_gold in tags:
        if one_gold in pred:
            TP += 1

    FP = len(pred) - TP
    FN = len(tags) - TP
    assert TP>=0
    assert FN>=0
    assert FP>=0

    new_dic_performance[task_key] = (TP, FN, FP)
    added_each_TP_FN_FP = add_new_relation(each_entity_TP_FN_FP, new_dic_performance)

    return added_each_TP_FN_FP

def get_each_class_P_R_F(accumulated_each_class_total_TP_FN_FP):
    dic_sub_task_P_R_F = {}
    for sub_task, (TP,FN,FP) in accumulated_each_class_total_TP_FN_FP.items():
        dic_sub_task_P_R_F[sub_task] = return_PRF(TP,FN,FP)
    return dic_sub_task_P_R_F

def get_each_corpus_micro_P_R_F(accumulated_each_class_total_TP_FN_FP, dic_sub_task_corpus, corpus_list):
    """
         if one sub-task occued in many corpus, when in multi-task metric, each corpus PRF are not total precise,
         which contain other entity mention with same entity type
    """

    dic_corpus_total_TP_FN_FP = {}
    for corpus in corpus_list:
        dic_corpus_total_TP_FN_FP.setdefault(corpus, [0,0,0])

    for sub_task, (TP,FN,FP) in accumulated_each_class_total_TP_FN_FP.items():
        corpus_list = dic_sub_task_corpus[sub_task]

        for corpus in corpus_list:
            dic_corpus_total_TP_FN_FP[corpus][0] = dic_corpus_total_TP_FN_FP[corpus][0] +TP
            dic_corpus_total_TP_FN_FP[corpus][1] = dic_corpus_total_TP_FN_FP[corpus][1] +FN
            dic_corpus_total_TP_FN_FP[corpus][2] = dic_corpus_total_TP_FN_FP[corpus][2] +FP

    dic_corpus_task_micro_P_R_F = {}
    for corpus, TP_FN_FP_list in dic_corpus_total_TP_FN_FP.items():
        dic_corpus_task_micro_P_R_F[corpus] = return_PRF(*TP_FN_FP_list)

    return dic_corpus_task_micro_P_R_F

def report_entity_span_PRF(list_batches_res, TAGS_fileds_list, dic_sub_task_corpus, improve_flag, corpus_list):
    task_key = "entity_span"
    entity_span_TP_FN_FP = {"entity_span": [0,0,0]}
    epoch_level_TP = 0
    epoch_level_FN = 0
    epoch_level_FP = 0
    dic_total_res = {}
    for one_batch_gold_list, one_batch_res_list in list_batches_res:
        for one_sent_pred, one_sent_gold in zip(one_batch_res_list, one_batch_gold_list):
            """get each classifier's TP_FN_FP, sent level """
            vocab = TAGS_fileds_list["entity_span"][1].vocab
            sentence_length = get_sent_len(one_sent_gold, vocab)
            targets_sent = [vocab.itos[int(i)] for i in one_sent_gold[:sentence_length]]
            pred_sent = [vocab.itos[int(i)] for i in one_sent_pred[:sentence_length]]
            tags_list = get_triple_O_seg(targets_sent)
            pred_list = get_triple_O_seg(pred_sent)
            tags_list = formatted_outpus(tags_list)
            pred_list = formatted_outpus(pred_list)
            each_classifier_TP_FN_FP = accumulated_each_class_TP_FN_FP(pred_list, tags_list, task_key, entity_span_TP_FN_FP)
            TP, FN, FP = each_classifier_TP_FN_FP[task_key]
            epoch_level_TP += TP
            epoch_level_FN += FN
            epoch_level_FP += FP
        dic_total_res[task_key] = (epoch_level_TP, epoch_level_FN, epoch_level_FP)

    epoch_micro_P, epoch_micro_R, epoch_micro_F = combine_all_class_for_total_PRF(dic_total_res)
    dic_corpus_task_micro_P_R_F = get_each_corpus_micro_P_R_F(each_classifier_TP_FN_FP, dic_sub_task_corpus, corpus_list)
    return epoch_micro_P, epoch_micro_R, epoch_micro_F, dic_corpus_task_micro_P_R_F, dic_total_res

def report_entity_type_PRF(list_batches_res, TAGS_fileds_list, dic_sub_task_corpus, corpus_list):
    sub_task_list = list(TAGS_fileds_list.keys())
    accumulated_each_class_total_TP_FN_FP = {}

    for one_batch_gold_dic, one_batch_pred_dic in list_batches_res:
        for one_sent_gold_dic, one_sent_pred_dic in zip(one_batch_gold_dic, one_batch_pred_dic):
            for sub_task in sub_task_list:
                if sub_task in dic_sub_task_corpus.keys():
                    accumulated_each_class_total_TP_FN_FP = accumulated_each_class_TP_FN_FP(one_sent_pred_dic[sub_task], one_sent_gold_dic[sub_task], sub_task, accumulated_each_class_total_TP_FN_FP)

    epoch_micro_P, epoch_micro_R, epoch_micro_F = combine_all_class_for_total_PRF(accumulated_each_class_total_TP_FN_FP)
    dic_sub_task_P_R_F = get_each_class_P_R_F(accumulated_each_class_total_TP_FN_FP)
    dic_corpus_task_micro_P_R_F = get_each_corpus_micro_P_R_F(accumulated_each_class_total_TP_FN_FP, dic_sub_task_corpus, corpus_list)

    return epoch_micro_P, epoch_micro_R, epoch_micro_F, dic_sub_task_P_R_F, dic_corpus_task_micro_P_R_F, accumulated_each_class_total_TP_FN_FP

def report_entity_span_and_type_PRF(list_batches_res, TAGS_fileds_list, dic_sub_task_corpus, improve_flag, corpus_list):
    accumulated_each_class_total_TP_FN_FP = {}
    sub_task_list = list(TAGS_fileds_list.keys())

    for one_batch_gold_dic, one_batch_pred_dic in list_batches_res:
        for sub_task in sub_task_list:
            if sub_task in dic_sub_task_corpus.keys():
                for one_sent_gold, one_sent_pred in zip(one_batch_gold_dic[sub_task], one_batch_pred_dic[sub_task]):
                    """get each classifier's TP_FN_FP, sent level """
                    vocab = [TAGS_fileds_list[sub_task]][0][1].vocab
                    sentence_length = get_sent_len(one_sent_gold, vocab)
                    targets_sentence = [vocab.itos[int(i)] for i in one_sent_gold[:sentence_length]]
                    tags_list = get_triple_O_seg(targets_sentence)
                    pred_list = improved_result(one_sent_pred) if improve_flag else one_sent_pred
                    accumulated_each_class_total_TP_FN_FP = accumulated_each_class_TP_FN_FP(pred_list, tags_list, sub_task, accumulated_each_class_total_TP_FN_FP)

    epoch_micro_P, epoch_micro_R, epoch_micro_F = combine_all_class_for_total_PRF(accumulated_each_class_total_TP_FN_FP)
    dic_sub_task_P_R_F = get_each_class_P_R_F(accumulated_each_class_total_TP_FN_FP)
    dic_corpus_task_micro_P_R_F = get_each_corpus_micro_P_R_F(accumulated_each_class_total_TP_FN_FP, dic_sub_task_corpus, corpus_list)

    return epoch_micro_P, epoch_micro_R, epoch_micro_F, dic_sub_task_P_R_F, dic_corpus_task_micro_P_R_F, accumulated_each_class_total_TP_FN_FP

def report_relation_PRF(list_batches_res, TAGS_Relation_fileds_dic, dic_sub_task_corpus, corpus_list):
    sub_task_list = list(TAGS_Relation_fileds_dic.keys())
    accumulated_each_class_total_TP_FN_FP = {}

    for one_batch_gold_list, one_batch_pred_list in list_batches_res:
        for one_sent_gold_dic, one_sent_pred_dic in zip(one_batch_gold_list, one_batch_pred_list):
            for sub_task in sub_task_list:
                if sub_task in dic_sub_task_corpus.keys():
                    accumulated_each_class_total_TP_FN_FP = accumulated_each_class_TP_FN_FP(one_sent_pred_dic[sub_task], one_sent_gold_dic[sub_task],
                                                                                            sub_task, accumulated_each_class_total_TP_FN_FP)

    dic_sub_task_P_R_F = get_each_class_P_R_F(accumulated_each_class_total_TP_FN_FP)
    epoch_micro_P, epoch_micro_R, epoch_micro_F = combine_all_class_for_total_PRF(accumulated_each_class_total_TP_FN_FP)
    dic_corpus_task_micro_P_R_F = get_each_corpus_micro_P_R_F(accumulated_each_class_total_TP_FN_FP, dic_sub_task_corpus, corpus_list)

    if epoch_micro_P>100 or epoch_micro_R>100 or epoch_micro_F>100:
        print(dic_sub_task_P_R_F)
        print(accumulated_each_class_total_TP_FN_FP)
        print(11)
    return epoch_micro_P, epoch_micro_R, epoch_micro_F, dic_sub_task_P_R_F, dic_corpus_task_micro_P_R_F, accumulated_each_class_total_TP_FN_FP

def report_performance(epoch, task_list, dic_loss, dic_batches_res, classifiers_dic, sep_corpus_file_dic, improve_flag, valid_flag):
    """
    :param dic_batches_res:  {"BIOES": [(one_batch_pred_sub_res, [batch.entity_span] ], "entity_type":[], "relation":[]}, len=batch_size
    """
    dic_PRF = {}
    dic_total_sub_task_P_R_F = {}
    dic_corpus_task_micro_P_R_F = {}
    dic_TP_FN_FP = {}

    dic_sub_task_corpus = {}
    corpus_list = []
    for corpus, task_dic in sep_corpus_file_dic.items():
        corpus_list.append(corpus)
        dic_corpus_task_micro_P_R_F.setdefault(corpus, {})
        for task, sub_task_list in task_dic.items():
            dic_corpus_task_micro_P_R_F[corpus].setdefault(task, [])
            for sub_task in sub_task_list:
                dic_sub_task_corpus.setdefault(sub_task, [])
                dic_sub_task_corpus[sub_task].append(corpus)

    if valid_flag=="train":
        print()
        print('Epoch: %1d, train average_loss: %2f' %(epoch, dic_loss["average"]))
    else:
        print("  validing ... ")

    if "entity_span" in task_list:
        entity_span_epoch_micro_P, \
        entity_span_epoch_micro_R, \
        entity_span_epoch_micro_F, \
        dic_corpus_task_micro_P_R_F_entity_span, \
        accumulated_each_class_total_TP_FN_FP = report_entity_span_PRF(
            dic_batches_res["entity_span"], classifiers_dic["entity_span"].TAGS_Types_fileds_dic, dic_sub_task_corpus, improve_flag, corpus_list)

        print('          entity_span : Loss: %.3f, P: %.3f, R: %.3f, F: %.3f '
              %(dic_loss["entity_span"], entity_span_epoch_micro_P, entity_span_epoch_micro_R, entity_span_epoch_micro_F))
        dic_PRF["entity_span"] = entity_span_epoch_micro_P, entity_span_epoch_micro_R, entity_span_epoch_micro_F
        dic_total_sub_task_P_R_F["entity_span"] = {"entity_span":(entity_span_epoch_micro_P, entity_span_epoch_micro_R, entity_span_epoch_micro_F)}
        for corpus, entity_span_PRF in dic_corpus_task_micro_P_R_F_entity_span.items():
            dic_corpus_task_micro_P_R_F[corpus]["entity_span"] = entity_span_PRF
        dic_TP_FN_FP["entity_span"] = accumulated_each_class_total_TP_FN_FP

    if "entity_type" in task_list:
        entity_type_epoch_micro_P, \
        entity_type_epoch_micro_R, \
        entity_type_epoch_micro_F, \
        dic_sub_task_P_R_F, \
        dic_corpus_task_micro_P_R_F_entity_type, \
        accumulated_each_class_total_TP_FN_FP = report_entity_type_PRF(dic_batches_res["entity_type"],
                                                                       classifiers_dic["entity_type"].TAGS_Types_fileds_dic,
                                                                       dic_sub_task_corpus, corpus_list)
        print('          entity_type : Loss: %.3f, P: %.3f, R: %.3f, F: %.3f'
              %(dic_loss["entity_type"], entity_type_epoch_micro_P, entity_type_epoch_micro_R, entity_type_epoch_micro_F))
        dic_PRF["entity_type"] = entity_type_epoch_micro_P, entity_type_epoch_micro_R, entity_type_epoch_micro_F
        dic_total_sub_task_P_R_F["entity_type"] = dic_sub_task_P_R_F
        for corpus, entity_span_PRF in dic_corpus_task_micro_P_R_F_entity_type.items():
            dic_corpus_task_micro_P_R_F[corpus]["entity_type"] = entity_span_PRF
        dic_TP_FN_FP["entity_type"] = accumulated_each_class_total_TP_FN_FP

    if "entity_span_and_type" in task_list:
        entity_span_and_type_epoch_micro_P, \
        entity_span_and_type_epoch_micro_R, \
        entity_span_and_type_epoch_micro_F, \
        dic_sub_task_P_R_F, \
        dic_corpus_task_micro_P_R_F_entity_span_and_type, \
        accumulated_each_class_total_TP_FN_FP = report_entity_span_and_type_PRF(dic_batches_res["entity_span_and_type"],
                                                                                classifiers_dic["entity_span_and_type"].TAGS_Types_fileds_dic,
                                                                                dic_sub_task_corpus, improve_flag, corpus_list)
        print('          entity_span_and_type : Loss: %.3f, P: %.3f, R: %.3f, F: %.3f'
              %(dic_loss["entity_span_and_type"], entity_span_and_type_epoch_micro_P, entity_span_and_type_epoch_micro_R, entity_span_and_type_epoch_micro_F))
        dic_PRF["entity_span_and_type"] =  entity_span_and_type_epoch_micro_P, entity_span_and_type_epoch_micro_R, entity_span_and_type_epoch_micro_F
        dic_total_sub_task_P_R_F["entity_span_and_type"] = dic_sub_task_P_R_F
        for corpus, entity_span_PRF in dic_corpus_task_micro_P_R_F_entity_span_and_type.items():
            dic_corpus_task_micro_P_R_F[corpus]["entity_span_and_type"] = entity_span_PRF
        dic_TP_FN_FP["entity_span_and_type"] = accumulated_each_class_total_TP_FN_FP

    if "relation" in task_list:
        relation_micro_P, \
        relation_micro_R, \
        relation_micro_F, \
        dic_sub_task_P_R_F, \
        dic_corpus_task_micro_P_R_F_relation, \
        accumulated_each_class_total_TP_FN_FP= report_relation_PRF(dic_batches_res["relation"],
                                                                   classifiers_dic["relation"].TAGS_Types_fileds_dic,
                                                                   dic_sub_task_corpus, corpus_list)
        print('          relation    : Loss: %.3f, P: %.3f, R: %.3f, F: %.3f \t\n\t\t\t'
              %(dic_loss["relation"], relation_micro_P, relation_micro_R, relation_micro_F))

        dic_PRF["relation"] = relation_micro_P, relation_micro_R, relation_micro_F
        dic_total_sub_task_P_R_F["relation"] = dic_sub_task_P_R_F
        for corpus, entity_span_PRF in dic_corpus_task_micro_P_R_F_relation.items():
            dic_corpus_task_micro_P_R_F[corpus]["relation"] = entity_span_PRF
        dic_TP_FN_FP["relation"] = accumulated_each_class_total_TP_FN_FP

    # print(dic_TP_FN_FP)
    return dic_PRF, dic_total_sub_task_P_R_F, dic_corpus_task_micro_P_R_F, dic_TP_FN_FP

def make_entity_span_test_data(TAGS_Entity_Span_fileds_dic):
    O_tag = TAGS_Entity_Span_fileds_dic['entity_span'][1].vocab.stoi["O"]
    B_tag = TAGS_Entity_Span_fileds_dic['entity_span'][1].vocab.stoi["B"]
    I_tag = TAGS_Entity_Span_fileds_dic['entity_span'][1].vocab.stoi["I"]
    E_tag = TAGS_Entity_Span_fileds_dic['entity_span'][1].vocab.stoi["E"]
    S_tag = TAGS_Entity_Span_fileds_dic['entity_span'][1].vocab.stoi["S"]

    make_one_gold_sent = torch.Tensor(1,50).fill_(O_tag)
    make_one_gold_sent[0][3] = S_tag
    make_one_gold_sent[0][12] = B_tag
    make_one_gold_sent[0][13] = I_tag
    make_one_gold_sent[0][14] = I_tag
    make_one_gold_sent[0][15] = E_tag
    make_one_gold_sent = make_one_gold_sent.tolist()

    make_one_pred_sent = torch.Tensor(1, 50).fill_(O_tag)
    make_one_pred_sent[0][3] = S_tag
    make_one_pred_sent[0][12] = B_tag
    make_one_pred_sent[0][13] = I_tag
    make_one_pred_sent[0][14] = E_tag
    make_one_pred_sent[0][25] = S_tag

    entity_span_list_batches_res = (make_one_gold_sent, make_one_pred_sent)
    return [entity_span_list_batches_res]

def make_entity_type_test_data(TAGS_Entity_Type_fileds_dic):
    gold_res = {}
    pre_res = {}
    key_list = []
    for key in TAGS_Entity_Type_fileds_dic.keys():
        gold_res[key] = []
        pre_res[key] = []
        key_list.append(key)

    # gold data
    gold_index_list = [0, 1]
    gold_pair_list = [
        ['[24, 25, 26, 27]', '[29, 30, 31]', '[45, 46, 47]', '[80, 81]'],
        ['[102, 103, 104]', '[5, 6, 7]'],
    ]
    for gold_index, gold_pair in zip(gold_index_list, gold_pair_list):
        gold_res[key_list[gold_index]].extend(gold_pair)

    # pre data
    pre_index_list = [0, 1]
    pre_pair_list = [
        ['[24, 25, 26, 27]', '[29, 30, 31]', '[45, 46, 47]', '[80, 81]'],
        ['[102, 103, 104]', '[5, 6, 7]'],
    ]
    for pre_index, pre_pair in zip(pre_index_list, pre_pair_list):
        pre_res[key_list[pre_index]].extend(pre_pair)

    return [([gold_res], [pre_res])]


def make_entity_span_and_type_test_data(TAGS_Entity_Span_And_Type_fileds_dic):
    vocab_list = {}
    for index, (sub_task, value) in enumerate(TAGS_Entity_Span_And_Type_fileds_dic.items()):
        vocab_list[sub_task] = {}
        for key in value[1].vocab.stoi.keys():
            vocab_list[sub_task][key] = value[1].vocab.stoi[key]

    gold_res = {}
    pre_res = {}
    for sub_task_index, sub_task in enumerate(vocab_list.keys()):
        if sub_task_index == 0:
            make_one_gold_sent = torch.Tensor(1, 50).fill_(vocab_list[sub_task]["O"])
            make_one_gold_sent[0][12] = vocab_list[sub_task]["B"]
            make_one_gold_sent[0][13] = vocab_list[sub_task]["I"]
            make_one_gold_sent[0][14] = vocab_list[sub_task]["E"]
            gold_res[sub_task] = make_one_gold_sent
            pre_res[sub_task] = [[[(3, 'S')], [(12, 'B'), (13, 'I'), (14, 'E')]]]
        elif sub_task_index == 1:
            make_one_gold_sent = torch.Tensor(1, 50).fill_(vocab_list[sub_task]["O"])
            make_one_gold_sent[0][12] = vocab_list[sub_task]["B"]
            make_one_gold_sent[0][13] = vocab_list[sub_task]["I"]
            make_one_gold_sent[0][14] = vocab_list[sub_task]["E"]
            gold_res[sub_task] = make_one_gold_sent
            pre_res[sub_task] = [[[(12, 'B'), (13, 'I'), (14, 'E')]]]
        else:
            make_one_gold_sent = torch.Tensor(1, 50).fill_(vocab_list[sub_task]["O"])
            gold_res[sub_task] = make_one_gold_sent
            pre_res[sub_task] = [[]]

    # make_one_gold_sent = make_one_gold_sent.tolist()
    return [(gold_res, pre_res)]

def make_relation_test_data(TAGS_Relation_fileds_dic):
    gold_res = {}
    pre_res = {}
    key_list = []
    for key in TAGS_Relation_fileds_dic.keys():
        gold_res[key] = []
        pre_res[key] = []
        key_list.append(key)

    # gold data
    gold_index_list = [0]
    gold_pair_list = [
        ['([24, 25, 26, 27], [29, 30, 31])', '([45, 46, 47], [80, 81, 82])'],
    ]
    for gold_index, gold_pair in zip(gold_index_list, gold_pair_list):
        gold_res[key_list[gold_index]].extend(gold_pair)

    # pre data
    pre_index_list = [0]
    pre_pair_list = [
        ['([24, 25, 26, 27], [29, 30, 31])', '([45, 46, 47], [80, 81, 82])'],
    ]
    for pre_index, pre_pair in zip(pre_index_list, pre_pair_list):
        pre_res[key_list[pre_index]].extend(pre_pair)

    return [([gold_res], [pre_res])]

if __name__ =="__main__":
    from transformers import *
    from data_loader import prepared_NER_data, prepared_RC_data, get_corpus_file_dic
    import argparse
    from my_modules import My_Entity_Span_Classifier, My_Entity_Type_Classifier, My_Entity_Span_And_Type_Classifier, My_Relation_Classifier, My_Bert_Encoder, My_Model

    parser = argparse.ArgumentParser(description="Bert Model")
    parser.add_argument('--GPU', default="3", type=str)
    parser.add_argument('--All_data', action='store_true', default=True)
    parser.add_argument('--BATCH_SIZE', default=8, type=int)

    parser.add_argument('--bert_model', default="base", type=str, help="base, large")
    parser.add_argument('--Task_list', default=["entity_span", "entity_type", "entity_span_and_type", "relation"], nargs='+', help=["entity_span", "entity_type", "entity_span_and_type", "relation"])
    parser.add_argument('--Task_weights_dic', default="{'entity_span':0.4, 'entity_type':0.1, 'entity_span_and_type':0.1, 'relation':0.4}", type=str)

    parser.add_argument('--Corpus_list', default=["ADE", "Twi_ADE", "DDI", "CPR"], nargs='+', help=["ADE", "Twi_ADE", "DDI", "CPR"])
    parser.add_argument('--Train_way', default="Multi_Task_Training", type=str)

    parser.add_argument('--Entity_Prep_Way', default="standard", type=str, help=["standard", "entitiy_type_marker"])
    parser.add_argument('--If_add_prototype', action='store_true', default=False) # True False

    parser.add_argument('--Average_Time', default=3, type=int)
    parser.add_argument('--EPOCH', default=100, type=int)
    parser.add_argument('--Min_train_performance_Report', default=25, type=int)
    parser.add_argument('--EARLY_STOP_NUM', default=25, type=int)

    parser.add_argument('--LR_max_bert', default=1e-5, type=float)
    parser.add_argument('--LR_min_bert', default=1e-6, type=float)
    parser.add_argument('--LR_max_entity_span', default=1e-4, type=float)
    parser.add_argument('--LR_min_entity_span', default=2e-6, type=float)
    parser.add_argument('--LR_max_entity_type', default=2e-5, type=float)
    parser.add_argument('--LR_min_entity_type', default=2e-6, type=float)
    parser.add_argument('--LR_max_entity_span_and_type', default=2e-5, type=float)
    parser.add_argument('--LR_min_entity_span_and_type', default=2e-6, type=float)
    parser.add_argument('--LR_max_relation', default=1e-4, type=float)
    parser.add_argument('--LR_min_relation', default=5e-6, type=float)
    parser.add_argument('--L2', default=1e-2, type=float)

    parser.add_argument('--Weight_Loss', action='store_true', default=True)
    parser.add_argument('--Loss', type=str, default="BCE", help=["BCE", "CE", "FSL"])
    parser.add_argument('--Min_weight', default=0.5, type=float)
    parser.add_argument('--Max_weight', default=5, type=float)
    parser.add_argument('--Tau', default=1.0, type=float)

    parser.add_argument('--Relation_input', default="entity_span", type=str, help=["entity_span", "entity_span_and_type"])
    parser.add_argument('--Only_relation', action='store_true', default=False)
    parser.add_argument('--Num_warmup_epoch', default=3, type=int)
    parser.add_argument('--Decay_epoch_num', default=40, type=float)
    parser.add_argument('--Min_valid_num', default=0, type=int)
    parser.add_argument('--Each_num_epoch_valid', default=1, type=int)
    parser.add_argument('--STOP_threshold', default=0, type=float)
    parser.add_argument('--Test_flag', action='store_true', default=False)
    parser.add_argument('--IF_CRF', action='store_true', default=False)
    parser.add_argument('--Improve_Flag', action='store_true', default=False)
    parser.add_argument('--Optim', default="AdamW", type=str, help=["AdamW"])
    parser.add_argument('--ID', default=0, type=int, help="Just for tensorboard reg")
    # AdamW  --LR_bert=5e-4 --LR_classfier=1e-3  70.285
    args = parser.parse_args()

    model_path = "../../../Data/embedding/biobert_base"
    # model_path = "../../Data/embedding/uncased_L-2_H-128_A-2"
    bert = BertModel.from_pretrained(model_path)
    tokenizer_NER = BertTokenizer.from_pretrained(model_path)
    tokenizer_RC = BertTokenizer.from_pretrained(model_path)

    if args.bert_model == "large":
        args.model_path = "../../../Data/embedding/biobert_large"
        args.Word_embedding_size = 1024
        args.Hidden_Size_Common_Encoder = args.Word_embedding_size
    elif args.bert_model == "base":
        args.model_path = "../../../Data/embedding/biobert_base"
        args.Word_embedding_size = 768
        args.Hidden_Size_Common_Encoder = args.Word_embedding_size

    dic_batches_res = {"entity_span":[], "entity_type":[], "relation":[], "entity_span_and_type":[]}
    all_data_flag = False
    task_list = ["entity_span", "entity_type", "entity_span_and_type", "relation"]
    corpus_list = ["Twi_ADE"]
    train_way = "Multi_Task_Training"
    device = "cpu"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_flag = "train"
    IF_CRF = False
    corpus_file_dic, sep_corpus_file_dic = get_corpus_file_dic(all_data_flag, corpus_list, task_list, "base")
    train_iterator_list = []
    valid_iterator_list = []
    test_iterator_list = []
    Average_Time_list = []


    my_entity_span_classifier = My_Entity_Span_Classifier(args, device)
    my_entity_type_classifier = My_Entity_Type_Classifier(args, device)
    my_entity_span_and_type_classifier = My_Entity_Span_And_Type_Classifier(args, device)
    my_relation_classifier = My_Relation_Classifier(args, tokenizer_RC, device)

    for corpus_name, (entity_type_num_list, relation_num_list, file_train_valid_test_list) in corpus_file_dic.items():
        NER_train_iterator, NER_valid_iterator_NER, NER_test_iterator_NER, NER_TOEKNS_fileds, TAGS_Entity_Span_fileds_dic, \
        TAGS_Entity_Type_fileds_dic, TAGS_Entity_Span_And_Type_fileds_dic, TAGS_sampled_entity_span_fileds_dic, TAGS_sep_entity_fileds_dic = \
            prepared_NER_data(args.BATCH_SIZE, device, tokenizer_NER, file_train_valid_test_list, entity_type_num_list)

        train_iterator_list.append(NER_train_iterator)
        valid_iterator_list.append(NER_test_iterator_NER)
        test_iterator_list.append(NER_test_iterator_NER)

        my_entity_span_classifier.create_classifers(TAGS_Entity_Span_fileds_dic)
        my_entity_type_classifier.create_classifers(TAGS_Entity_Type_fileds_dic, TAGS_sep_entity_fileds_dic)
        my_entity_span_and_type_classifier.create_classifers(TAGS_Entity_Span_And_Type_fileds_dic)

        if "relation" in args.Task_list:
            RC_train_iterator, RC_valid_iterator, RC_test_iterator, RC_TOEKNS_fileds, TAGS_Relation_pair_fileds_dic, TAGS_sampled_entity_span_fileds_dic = \
                prepared_RC_data(args.BATCH_SIZE, device, tokenizer_RC, file_train_valid_test_list, relation_num_list)
            my_relation_classifier.create_classifers(TAGS_Relation_pair_fileds_dic, TAGS_sampled_entity_span_fileds_dic, TAGS_Entity_Type_fileds_dic)

            train_iterator_list.append(RC_train_iterator)
            valid_iterator_list.append(RC_test_iterator)
            test_iterator_list.append(RC_test_iterator)

        dic_loss = {"average":0, "entity_span":0, "entity_type":0, "entity_span_and_type":0, "relation":0,}


        dic_batches_res["entity_span"] = make_entity_span_test_data(TAGS_Entity_Span_fileds_dic )
        dic_batches_res["entity_type"] = make_entity_type_test_data(TAGS_Entity_Type_fileds_dic)
        dic_batches_res["entity_span_and_type"] = make_entity_span_and_type_test_data(TAGS_Entity_Span_And_Type_fileds_dic)
        dic_batches_res["relation"] = make_relation_test_data(TAGS_Relation_pair_fileds_dic)


        classifiers_dic = dict(zip(["entity_span", "entity_type", "entity_span_and_type", "relation"], [my_entity_span_classifier, my_entity_type_classifier, my_entity_span_and_type_classifier, my_relation_classifier]))
        dic_PRF, dic_total_sub_task_P_R_F, dic_corpus_task_micro_P_R_F, dic_TP_FN_FP = report_performance(0, task_list, dic_loss, dic_batches_res, classifiers_dic, sep_corpus_file_dic, False, valid_flag)
        print(dic_TP_FN_FP)
















