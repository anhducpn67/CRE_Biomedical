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
    return P * 100, R * 100, F * 100


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
    assert TP >= 0
    assert FN >= 0
    assert FP >= 0

    new_dic_performance[task_key] = (TP, FN, FP)
    added_each_TP_FN_FP = add_new_relation(each_entity_TP_FN_FP, new_dic_performance)

    return added_each_TP_FN_FP


def get_each_class_P_R_F(accumulated_each_class_total_TP_FN_FP):
    dic_sub_task_P_R_F = {}
    for sub_task, (TP, FN, FP) in accumulated_each_class_total_TP_FN_FP.items():
        dic_sub_task_P_R_F[sub_task] = return_PRF(TP, FN, FP)
    return dic_sub_task_P_R_F


def get_each_corpus_micro_P_R_F(accumulated_each_class_total_TP_FN_FP, dic_sub_task_corpus, corpus_list):
    """
         if one sub-task occued in many corpus, when in multi-task metric, each corpus PRF are not total precise,
         which contain other entity mention with same entity type
    """

    dic_corpus_total_TP_FN_FP = {}
    for corpus in corpus_list:
        dic_corpus_total_TP_FN_FP.setdefault(corpus, [0, 0, 0])

    for sub_task, (TP, FN, FP) in accumulated_each_class_total_TP_FN_FP.items():
        corpus_list = dic_sub_task_corpus[sub_task]

        for corpus in corpus_list:
            dic_corpus_total_TP_FN_FP[corpus][0] = dic_corpus_total_TP_FN_FP[corpus][0] + TP
            dic_corpus_total_TP_FN_FP[corpus][1] = dic_corpus_total_TP_FN_FP[corpus][1] + FN
            dic_corpus_total_TP_FN_FP[corpus][2] = dic_corpus_total_TP_FN_FP[corpus][2] + FP

    dic_corpus_task_micro_P_R_F = {}
    for corpus, TP_FN_FP_list in dic_corpus_total_TP_FN_FP.items():
        dic_corpus_task_micro_P_R_F[corpus] = return_PRF(*TP_FN_FP_list)

    return dic_corpus_task_micro_P_R_F


def report_relation_PRF(list_batches_res, TAGS_Relation_fileds_dic, dic_sub_task_corpus, corpus_list):
    sub_task_list = list(TAGS_Relation_fileds_dic.keys())
    accumulated_each_class_total_TP_FN_FP = {}

    for one_batch_gold_list, one_batch_pred_list in list_batches_res:
        for one_sent_gold_dic, one_sent_pred_dic in zip(one_batch_gold_list, one_batch_pred_list):
            for sub_task in sub_task_list:
                if sub_task in dic_sub_task_corpus.keys():
                    accumulated_each_class_total_TP_FN_FP = accumulated_each_class_TP_FN_FP(one_sent_pred_dic[sub_task],
                                                                                            one_sent_gold_dic[sub_task],
                                                                                            sub_task,
                                                                                            accumulated_each_class_total_TP_FN_FP)

    dic_sub_task_P_R_F = get_each_class_P_R_F(accumulated_each_class_total_TP_FN_FP)
    epoch_micro_P, epoch_micro_R, epoch_micro_F = combine_all_class_for_total_PRF(accumulated_each_class_total_TP_FN_FP)
    dic_corpus_task_micro_P_R_F = get_each_corpus_micro_P_R_F(accumulated_each_class_total_TP_FN_FP,
                                                              dic_sub_task_corpus, corpus_list)

    if epoch_micro_P > 100 or epoch_micro_R > 100 or epoch_micro_F > 100:
        print(dic_sub_task_P_R_F)
        print(accumulated_each_class_total_TP_FN_FP)
        print(11)
    return epoch_micro_P, epoch_micro_R, epoch_micro_F, dic_sub_task_P_R_F, dic_corpus_task_micro_P_R_F, accumulated_each_class_total_TP_FN_FP


def report_performance(corpus_name, epoch, dic_loss, dic_batches_res, relation_classifier, sep_corpus_file_dic, valid_flag):
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

    if valid_flag == "train":
        print()
        print(corpus_name)
        print('Epoch: %1d, train average_loss: %2f' % (epoch, dic_loss["average"]))
    else:
        print(corpus_name)
        print("  validing ... ")

    relation_micro_P, \
        relation_micro_R, \
        relation_micro_F, \
        dic_sub_task_P_R_F, \
        dic_corpus_task_micro_P_R_F_relation, \
        accumulated_each_class_total_TP_FN_FP = report_relation_PRF(dic_batches_res["relation"],
                                                                    relation_classifier.TAGS_Types_fields_dic,
                                                                    dic_sub_task_corpus, corpus_list)
    print('          relation    : Loss: %.3f, P: %.3f, R: %.3f, F: %.3f \t\n\t\t\t'
          % (dic_loss["relation"], relation_micro_P, relation_micro_R, relation_micro_F))

    dic_PRF["relation"] = relation_micro_P, relation_micro_R, relation_micro_F
    dic_total_sub_task_P_R_F["relation"] = dic_sub_task_P_R_F
    for corpus, entity_span_PRF in dic_corpus_task_micro_P_R_F_relation.items():
        dic_corpus_task_micro_P_R_F[corpus]["relation"] = entity_span_PRF
    dic_TP_FN_FP["relation"] = accumulated_each_class_total_TP_FN_FP

    return dic_PRF, dic_total_sub_task_P_R_F, dic_corpus_task_micro_P_R_F, dic_TP_FN_FP
