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


# def calc_corpus_micro_P_R_F(relation_TP_FN_FP, corpus_information):
#     dic_corpus_total_TP_FN_FP = {}
#     for corpus in corpus_information.keys():
#         dic_corpus_total_TP_FN_FP[corpus] = [0, 0, 0]
#         for relation in corpus_information[corpus]["relation_list"]:
#             TP, FN, FP = relation_TP_FN_FP[relation]
#             dic_corpus_total_TP_FN_FP[corpus][0] = dic_corpus_total_TP_FN_FP[corpus][0] + TP
#             dic_corpus_total_TP_FN_FP[corpus][1] = dic_corpus_total_TP_FN_FP[corpus][1] + FN
#             dic_corpus_total_TP_FN_FP[corpus][2] = dic_corpus_total_TP_FN_FP[corpus][2] + FP
#
#     corpus_micro_P_R_F1 = {}
#     for corpus, (TP, FN, FP) in dic_corpus_total_TP_FN_FP.items():
#         corpus_micro_P_R_F1[corpus] = calc_P_R_F1(TP, FN, FP)
#
#     return corpus_micro_P_R_F1


def report_relation_PRF(list_batch_res, relation_list):
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
    return micro_P, micro_R, micro_F1, relation_P_R_F1, relation_TP_FN_FP


def report_performance(corpus_name, epoch, dic_loss, dic_batches_res, relation_list, valid_flag):
    if valid_flag == "train":
        print(corpus_name)
        print('Epoch: %1d, train average_loss: %2f' % (epoch, dic_loss["average"]))
    else:
        print(corpus_name)
        print("  validing ... ")

    micro_P, micro_R, micro_F1, relation_P_R_F1, relation_TP_FN_FP = report_relation_PRF(dic_batches_res["relation"],
                                                                                         relation_list)
    print('          P: %.3f, R: %.3f, F1: %.3f \t\n\t\t\t'
          % (micro_P, micro_R, micro_F1))

    micro_P_R_F1 = (micro_P, micro_R, micro_F1)

    return micro_P_R_F1, relation_P_R_F1, relation_TP_FN_FP


def record_detail_performance(relation_P_R_F1, micro_P_R_F1, file_detail_performance,
                              relation_TP_FN_FP, corpus_information, corpus_list):
    with open(file_detail_performance, "w") as f:
        f.write("total performance: " + "\n")
        f.write("\tMicro-P: " + str(round(micro_P_R_F1[0], 3)) + "\n")
        f.write("\tMicro-R: " + str(round(micro_P_R_F1[1], 3)) + "\n")
        f.write("\tMicro-F1: " + str(round(micro_P_R_F1[2], 3)) + "\n")
        f.write("\n")
        f.write("============================================================\n")

        for corpus in corpus_list:
            f.write(corpus + "\n")
            for relation in corpus_information[corpus]["relation_list"]:
                f.write("\t %-16s: %-25s \t TP_FN_FP: %-5s \n" % (str(relation),
                                                                  str([round(i, 3) for i in relation_P_R_F1[relation]]),
                                                                  str(relation_TP_FN_FP[relation])))
            f.write("\n")
            f.write("============================================================\n")
