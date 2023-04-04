import torch
import torch.nn as nn
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
from allennlp.nn import util as nn_util


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


def recored_detail_performance(epoch, dic_total_sub_task_P_R_F, dic_valid_PRF, file_detail_performance,
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


def get_loss(targets, TAGS, IF_Weight_loss, Weight_Max, relation_tag_weight):
    if IF_Weight_loss:
        # auto weights or fix weights
        if Weight_Max != 0:
            relation_tag_weight_tensor = get_adaptive_weight_tensor(targets, TAGS, Weight_Max)

            criterion = nn.CrossEntropyLoss(ignore_index=TAGS['[PAD]'],
                                            weight=relation_tag_weight_tensor)
        else:
            list_weight = [relation_tag_weight] * len(TAGS)
            list_weight[TAGS['O']] = 1
            list_weight[TAGS['[PAD]']] = 1
            relation_tag_weight_tensor = torch.Tensor(list_weight).cuda()
            criterion = nn.CrossEntropyLoss(ignore_index=TAGS['[PAD]'],
                                            weight=relation_tag_weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=TAGS['[PAD]'])

    return criterion


def get_adaptive_weight_tensor(targets, TAGS, Weight_Max):
    target = torch.argmax(targets, 2).view(-1).tolist()
    class_dic = dict(Counter(target))
    total_num = 0
    for key, values in class_dic.items():
        if key != TAGS["[PAD]"]:
            total_num += values

    dic_weight = {}
    for key, values in class_dic.items():
        weight = total_num / class_dic[key]
        dic_weight[key] = weight

    list_weight0 = [1] * len(TAGS)
    for i, v in enumerate(list_weight0):
        if i in dic_weight.keys():
            list_weight0[i] = dic_weight[i]

    minmaxScaler = MinMaxScaler(feature_range=(1, Weight_Max))
    list_weight1 = minmaxScaler.fit_transform(np.array(list_weight0).reshape(-1, 1))

    relation_tag_weight_tensor = torch.Tensor(list_weight1).cuda()
    return relation_tag_weight_tensor


def log_gradient_updates(writer, model, epoch) -> None:
    param_updates = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
    }

    for name, param in model.named_parameters():
        update_norm = torch.norm(param_updates[name].view(-1))
        param_norm = torch.norm(param.view(-1)).cpu()
        writer.add_scalars('Model/' + name, {
            "gradient_update/" + name: update_norm / (param_norm + nn_util.tiny_value_of_dtype(param_norm.dtype))},
                           epoch)


def log_parameter_and_gradient_statistics(model, writer, epoch) -> None:
    """
    Send the mean and std of all parameters and gradients to tensorboard, as well
    as logging the average gradient norm.
    """

    for name, param in model.named_parameters():
        if param.data.numel() > 0:
            writer.add_scalars("Model/" + name, {"parameter_mean/" + name: param.data.mean().item()}, epoch)
        if param.data.numel() > 1:
            writer.add_scalars("Model/" + name, {"parameter_std/" + name: param.data.std().item()}, epoch)
        if param.grad is not None:
            if param.grad.is_sparse:
                grad_data = param.grad.data._values()
            else:
                grad_data = param.grad.data

            # skip empty gradients
            if torch.prod(torch.tensor(grad_data.shape)).item() > 0:
                writer.add_scalars("Model/" + name, {"gradient_mean/" + name: grad_data.mean()}, epoch)
                if grad_data.numel() > 1:
                    writer.add_scalars("Model/" + name, {"gradient_std/" + name: grad_data.std()}, epoch)

    # norm of gradients
    parameters_to_clip = [p for p in model.parameters() if p.grad is not None]
    batch_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in parameters_to_clip]))
    writer.add_scalars("Model", {"gradient_norm": batch_grad_norm}, epoch)


class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass