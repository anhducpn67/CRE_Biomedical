import sys


def print_execute_time(func):
    from time import time
    num = 0

    def wrapper(*args, **kwargs):
        nonlocal num
        num += 1

        start = time()
        func_return = func(*args, **kwargs)
        end = time()
        opera_time = end - start
        if opera_time > 3600:
            print()
            print(f'{func.__name__}() execute time: {round(opera_time / 3600, 3)}h, and run: {num} time.')
        elif opera_time > 60:
            print()
            print(f'{func.__name__}() execute time: {round(opera_time / 60, 3)}m, and run: {num} time.')
        else:
            print()
            print(f'{func.__name__}() execute time: {round(opera_time, 3)}s, and run: {num} time.')

        return func_return

    return wrapper


def record_detail_performance(epoch, dic_total_sub_task_P_R_F, dic_valid_PRF, file_detail_performance,
                              dic_corpus_task_micro_P_R_F, dic_TP_FN_FP, corpus_file_dic, task_list, corpus_list,
                              average_Time):
    with open(file_detail_performance, "w") as f:
        f.write("only record average_Time in : " + str(average_Time) + "\n")
        f.write("epoch: " + str(epoch) + "\n")
        f.write("total performance: " + "\n")
        for k, v in dic_valid_PRF.items():
            f.write("\t" + str(k) + ": " + str([round(i, 3) for i in v]) + "\n")
        f.write("\n")
        f.write("============================================================")

        for corpus, corpus_dic in corpus_file_dic.items():
            if corpus in corpus_list:
                f.write("\n")
                f.write(corpus + "\n")
                for need_task in task_list:
                    for sub_task in corpus_dic[need_task]:
                        try:
                            if sub_task != "entity_span":
                                PRF = dic_total_sub_task_P_R_F[need_task][sub_task]
                                f.write("\t %-16s: %-25s \t TP_FN_FP: %-5s \n" % (str(sub_task),
                                                                                  str([round(i, 3) for i in PRF]),
                                                                                  str(dic_TP_FN_FP[need_task][sub_task])))
                        except KeyError:
                            pass
                    f.write("\n")
                f.write("\n")

                f.write("corpus level PRF (micro): \n")
                for task, PRF in dic_corpus_task_micro_P_R_F[corpus].items():
                    if task == "relation":
                        f.write("\t" + task + ": " + str([round(i, 3) for i in PRF]) + "\n")
                f.write("============================================================")


def get_sent_len(one_sent_tags, entity_BIOES_TAGS_filed):
    sentence_length = 0
    for tags in one_sent_tags:
        if int(tags) != entity_BIOES_TAGS_filed.vocab["[PAD]"]:
            sentence_length += 1
    return sentence_length


class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
