def calc_micro_P_R_F1(relation_TP_FN_FP):
    total_TP = 0
    total_FN = 0
    total_FP = 0
    for relation, (TP, FN, FP) in relation_TP_FN_FP.items():
        total_TP += TP
        total_FN += FN
        total_FP += FP

    micro_P, micro_R, micro_F1 = calc_P_R_F1(total_TP, total_FN, total_FP)

    return micro_P, micro_R, micro_F1


def calc_P_R_F1(TP, FN, FP):
    P = TP / (TP + FP) if (TP + FP) != 0 else 0
    R = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) != 0 else 0
    return P * 100, R * 100, F1 * 100


def accumulated_relation_TP_FN_FP(pred, gold, relation, relation_TP_FN_FP):
    TP = 0
    for one_gold in gold:
        if one_gold in pred:
            TP += 1

    FP = len(pred) - TP
    FN = len(gold) - TP
    assert TP >= 0
    assert FN >= 0
    assert FP >= 0

    if relation not in relation_TP_FN_FP.keys():
        relation_TP_FN_FP[relation] = [TP, FN, FP]
    else:
        relation_TP_FN_FP[relation][0] += TP
        relation_TP_FN_FP[relation][1] += FN
        relation_TP_FN_FP[relation][2] += FP
    return relation_TP_FN_FP


def calc_relation_P_R_F1(relation_TP_FN_FP):
    relation_P_R_F1 = {}
    for relation, (TP, FN, FP) in relation_TP_FN_FP.items():
        relation_P_R_F1[relation] = calc_P_R_F1(TP, FN, FP)
    return relation_P_R_F1


def calc_corpus_micro_P_R_F(relation_TP_FN_FP, dic_sub_task_corpus, corpus_list):
    """
         if one sub-task occued in many corpus, when in multi-task metric, each corpus PRF are not total precise,
         which contain other entity mention with same entity type
    """

    dic_corpus_total_TP_FN_FP = {}
    for corpus in corpus_list:
        dic_corpus_total_TP_FN_FP.setdefault(corpus, [0, 0, 0])

    for relation, (TP, FN, FP) in relation_TP_FN_FP.items():
        corpus_list = dic_sub_task_corpus[relation]

        for corpus in corpus_list:
            dic_corpus_total_TP_FN_FP[corpus][0] = dic_corpus_total_TP_FN_FP[corpus][0] + TP
            dic_corpus_total_TP_FN_FP[corpus][1] = dic_corpus_total_TP_FN_FP[corpus][1] + FN
            dic_corpus_total_TP_FN_FP[corpus][2] = dic_corpus_total_TP_FN_FP[corpus][2] + FP

    dic_corpus_task_micro_P_R_F = {}
    for corpus, TP_FN_FP_list in dic_corpus_total_TP_FN_FP.items():
        dic_corpus_task_micro_P_R_F[corpus] = calc_P_R_F1(*TP_FN_FP_list)

    return dic_corpus_task_micro_P_R_F


def report_relation_PRF(list_batch_res, TAGS_Relation_fields_dic, dic_sub_task_corpus, corpus_list):
    relation_list = list(TAGS_Relation_fields_dic.keys())
    relation_TP_FN_FP = {}

    for batch_gold, batch_pred in list_batch_res:
        for sent_gold, sent_pred in zip(batch_gold, batch_pred):
            for relation in relation_list:
                relation_TP_FN_FP = accumulated_relation_TP_FN_FP(sent_pred[relation],
                                                                  sent_gold[relation],
                                                                  relation,
                                                                  relation_TP_FN_FP)

    relation_P_R_F1 = calc_relation_P_R_F1(relation_TP_FN_FP)
    micro_P, micro_R, micro_F1 = calc_micro_P_R_F1(relation_TP_FN_FP)
    corpus_micro_P_R_F1 = calc_corpus_micro_P_R_F(relation_TP_FN_FP, dic_sub_task_corpus, corpus_list)

    return micro_P, micro_R, micro_F1, relation_P_R_F1, corpus_micro_P_R_F1, relation_TP_FN_FP


def report_performance(corpus_name, epoch, dic_loss, dic_batches_res, relation_classifier, sep_corpus_file_dic, valid_flag):
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

    micro_P, \
        micro_R, \
        micro_F1, \
        relation_P_R_F1, \
        dic_corpus_task_micro_P_R_F_relation, \
        accumulated_each_class_total_TP_FN_FP = report_relation_PRF(dic_batches_res["relation"],
                                                                    relation_classifier.TAGS_Types_fields_dic,
                                                                    dic_sub_task_corpus, corpus_list)
    print('          relation    : Loss: %.3f, P: %.3f, R: %.3f, F: %.3f \t\n\t\t\t'
          % (dic_loss["relation"], micro_P, micro_R, micro_F1))

    micro_P_R_F1 = (micro_P, micro_R, micro_F1)
    dic_total_sub_task_P_R_F["relation"] = relation_P_R_F1
    for corpus, entity_span_PRF in dic_corpus_task_micro_P_R_F_relation.items():
        dic_corpus_task_micro_P_R_F[corpus]["relation"] = entity_span_PRF
    dic_TP_FN_FP["relation"] = accumulated_each_class_total_TP_FN_FP

    return micro_P_R_F1, dic_total_sub_task_P_R_F, dic_corpus_task_micro_P_R_F, dic_TP_FN_FP
