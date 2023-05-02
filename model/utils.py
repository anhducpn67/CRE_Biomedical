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


def get_sent_len(one_sent_tags, TAGS_field):
    sentence_length = 0
    for tags in one_sent_tags:
        if int(tags) != TAGS_field.vocab["[PAD]"]:
            sentence_length += 1
    return sentence_length


class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
