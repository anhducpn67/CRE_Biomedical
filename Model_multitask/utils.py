import torch
import torch.nn as nn
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
from metric import get_triple_O_seg, formatted_outpus
import json
from allennlp.nn import util as nn_util
from torch.optim.lr_scheduler import LambdaLR


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


def get_entity_type_rep_dic(train_iterator_list, valid_iterator_list, my_model, dim_num, device):
    my_entity_type_sub_task_list = my_model.my_entity_type_classifier.my_entity_type_sub_task_list
    # word_embeddings = my_model.bert_NER.bert.embeddings.word_embeddings.weight
    word_embeddings = my_model.bert_RC.bert.embeddings.word_embeddings.weight
    TAGS_type_dic = my_model.my_entity_type_classifier.TAGS_Types_fileds_dic

    entity_type_rep_dic = {}
    for entity_type in my_entity_type_sub_task_list:
        assert TAGS_type_dic[entity_type][1].vocab.itos[0] == "[PAD]"
        entity_type_rep_dic.setdefault(entity_type, [torch.FloatTensor([0] * word_embeddings.shape[1]).to(device)])
        for train_iterator, valid_iterator in zip(train_iterator_list, valid_iterator_list):
            for train_batch, valid_batch in zip(train_iterator, valid_iterator):
                train_batch_tokens = train_batch.tokens
                valid_batch_tokens = valid_batch.tokens
                if hasattr(valid_batch, entity_type) and hasattr(train_batch, entity_type):
                    for sent_index, (train_sent_index_v, valid_sent_index_v) in enumerate(
                            zip(getattr(train_batch, entity_type), getattr(valid_batch, entity_type))):
                        for train_token_index, valid_token_index in zip(train_sent_index_v, valid_sent_index_v):
                            if train_token_index != 0:
                                entity_span = eval(TAGS_type_dic[entity_type][1].vocab.itos[train_token_index])
                                tokens_index = train_batch_tokens[sent_index][entity_span[0]:entity_span[-1] + 1]
                                entity_embed = sum([word_embeddings[token_index] for token_index in tokens_index])
                                entity_type_rep_dic[entity_type].append(entity_embed)
                            if valid_token_index != 0:
                                entity_span = eval(TAGS_type_dic[entity_type][1].vocab.itos[valid_token_index])
                                tokens_index = valid_batch_tokens[sent_index][entity_span[0]:entity_span[-1] + 1]
                                entity_embed = sum([word_embeddings[token_index] for token_index in tokens_index])
                                entity_type_rep_dic[entity_type].append(entity_embed)

    max_pool = nn.AdaptiveMaxPool1d(dim_num)
    for k, v in entity_type_rep_dic.items():
        entity_type_rep_dic[k] = max_pool(torch.mean(torch.stack(v), dim=0).data.reshape(1, 1, -1)).squeeze()

    entity_type_rep_dic["None"] = torch.FloatTensor([0] * dim_num).to(device)
    return entity_type_rep_dic


def my_schedulermy_scheduler(optimizer, num_warmup_steps, num_training_steps, lr_end=1e-6, last_epoch=-1, power=1):
    lr_init = optimizer.defaults["lr"]
    assert lr_init > lr_end, f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_epoch_level_sent_res(epoch_sent_list, max_len, tokenizer, device):
    new_epoch_sent_list = []
    for batch_tensor in epoch_sent_list:
        temp_list = []
        for sent_sensor in batch_tensor:
            need_dim = max_len - sent_sensor.shape[0]
            pad_pred_tensor = torch.ones(need_dim).to(device) * tokenizer.vocab["[PAD]"]
            sent_sensor = torch.cat([sent_sensor, pad_pred_tensor], dim=0)
            temp_list.append(sent_sensor)
        new_batch_tensor = torch.stack(temp_list, dim=0)
        new_epoch_sent_list.append(new_batch_tensor)
    epoch_sent = torch.cat(new_epoch_sent_list, dim=0)
    return epoch_sent


def get_epoch_level_res(epoch_pred_list, epoch_tags_list, max_len, TAGS_filed, device, IF_CRF=False):
    """
    :param epoch_pred_list: padding to max len for all batch, aiming at counting PRF for one epoch
    :param epoch_tags_list: padding to max len for all batch, aiming at counting PRF for one epoch
    :param max_len: for all batch
    :param TAGS_filed:
    :param tokenizer:
    :param device:
    :return:
    """

    new_epoch_pred_list = []
    new_epoch_tags_list = []

    for batch_tensor in epoch_pred_list:
        temp_list = []
        for sent_sensor in batch_tensor:
            need_dim = max_len - sent_sensor.shape[0]
            if IF_CRF:
                pad_pred_tensor = torch.ones((need_dim, 1)).to(device) * TAGS_filed.vocab["[PAD]"]
            else:
                pad_pred_tensor = torch.ones((need_dim, len(TAGS_filed.vocab))).to(device) * TAGS_filed.vocab["[PAD]"]
            sent_sensor = torch.cat([sent_sensor, pad_pred_tensor], dim=0)
            temp_list.append(sent_sensor)
        new_batch_tensor = torch.stack(temp_list, dim=0)
        new_epoch_pred_list.append(new_batch_tensor)

    for batch_tensor in epoch_tags_list:
        temp_list = []
        for sent_sensor in batch_tensor:
            need_dim = max_len - sent_sensor.shape[0]
            pad_tag_tensor = torch.ones(need_dim).to(device) * TAGS_filed.vocab.stoi["[PAD]"]
            sent_sensor = torch.cat([sent_sensor, pad_tag_tensor], dim=0)
            temp_list.append(sent_sensor)
        new_batch_tensor = torch.stack(temp_list, dim=0)
        new_epoch_tags_list.append(new_batch_tensor)

    epoch_pred = torch.cat(new_epoch_pred_list, dim=0)
    epoch_tags = torch.cat(new_epoch_tags_list, dim=0)

    return epoch_pred, epoch_tags


def record_pred_str_res(task_list, file, dic_batches_res, average_num, id_token_map,
                        # TAGS_Entity_Span_fileds_dic,
                        # TAGS_Entity_Type_fileds_dic,
                        # TAGS_Entity_Span_And_Type_fileds_dic,
                        # TAGS_Relation_fileds_dic,
                        classifiers_dic,
                        IF_CRF):
    write_flag = "w"
    sent_id = 0
    with open(file, write_flag) as f:
        batch_num = len(dic_batches_res["ID_list"])
        for batch_index in range(batch_num):
            for sent_index, sent_ID in enumerate(dic_batches_res["ID_list"][batch_index]):
                res_dic = {}
                sen_len = 0
                sent_id = sent_id + 1
                res_dic["<sentence>-NO."] = sent_id
                res_dic["ID"] = int(sent_ID)

                sent_str = ""
                token_list = []
                for word in dic_batches_res["tokens_list"][batch_index][sent_index]:
                    token = id_token_map(int(word))
                    token_list.append(token)
                    sen_len = sen_len + 1
                    if token == '[PAD]':
                        break
                    sent_str = sent_str + " " + token
                res_dic["Sentence"] = sent_str.replace(" ##", "")

                if "entity_span" in task_list:
                    vocab = classifiers_dic["entity_span"].TAGS_Types_fileds_dic["entity_span"][1].vocab.itos

                    sent_entity_span_gold = dic_batches_res["entity_span"][batch_index][0][sent_index]
                    sent_entity_span_gold = [vocab[item] for item in sent_entity_span_gold[:sen_len]]
                    sent_entity_span_gold = get_triple_O_seg(sent_entity_span_gold)
                    sent_entity_span_gold = [str(i) for i in sent_entity_span_gold]

                    sent_entity_span_pred = dic_batches_res["entity_span"][batch_index][1][sent_index]
                    sent_entity_span_pred = [vocab[item] for item in sent_entity_span_pred[:sen_len]]
                    sent_entity_span_pred = get_triple_O_seg(sent_entity_span_pred)
                    sent_entity_span_pred = [str(i) for i in sent_entity_span_pred]

                    res_dic["entity_span"] = {"gold": sent_entity_span_gold, "pred": sent_entity_span_pred}

                if "entity_type" in task_list:
                    gold_all_sub_task_dic = dic_batches_res["entity_type"][batch_index][0][sent_index]
                    pred_all_sub_task_dic = dic_batches_res["entity_type"][batch_index][1][sent_index]
                    res_dic["entity_type"] = {"gold_all_sub": str(gold_all_sub_task_dic),
                                              "pred_all_sub": str(pred_all_sub_task_dic)}

                if "entity_span_and_type" in task_list:
                    TAGS_Entity_Span_And_Type_fileds_dic = classifiers_dic["entity_span_and_type"].TAGS_Types_fileds_dic
                    sent_relation_gold_all_task = dic_batches_res["entity_span_and_type"][batch_index][0]
                    sent_relation_pred_all_task = dic_batches_res["entity_span_and_type"][batch_index][1]
                    sub_task_list = list(TAGS_Entity_Span_And_Type_fileds_dic.keys())
                    dic_gold_entity_span_and_type_res = {}
                    dic_pred_entity_span_and_type_res = {}
                    for sub_task in sub_task_list:
                        dic_gold_entity_span_and_type_res.setdefault(sub_task, [])
                        dic_pred_entity_span_and_type_res.setdefault(sub_task, [])
                        vocab = TAGS_Entity_Span_And_Type_fileds_dic[sub_task][1].vocab.itos
                        sent_entity_type_gold = sent_relation_gold_all_task[sub_task][sent_index]
                        sent_entity_type_gold = [vocab[item] for item in sent_entity_type_gold[:sen_len]]
                        sent_entity_type_gold = get_triple_O_seg(sent_entity_type_gold)
                        sent_entity_type_pred = sent_relation_pred_all_task[sub_task][sent_index]
                        dic_gold_entity_span_and_type_res[sub_task] = sent_entity_type_gold
                        dic_pred_entity_span_and_type_res[sub_task] = sent_entity_type_pred
                    res_dic["entity_span_and_type"] = {"gold_all_sub": str(dic_gold_entity_span_and_type_res),
                                                       "pred_all_sub": str(dic_pred_entity_span_and_type_res)}

                if "relation" in task_list:
                    gold_all_sub_task_dic = dic_batches_res["relation"][batch_index][0][sent_index]
                    pred_all_sub_task_dic = dic_batches_res["relation"][batch_index][1][sent_index]
                    res_dic["relation"] = {"gold_all_sub": str(gold_all_sub_task_dic),
                                           "pred_all_sub": str(pred_all_sub_task_dic)}

                f.write(json.dumps(res_dic))
                f.write("\n")


def record_each_performance(file, Task_list, dic_valid_PRF_list):
    if str(sys.argv[1:]) != "[]":
        dic_temp_P = {}
        dic_temp_R = {}
        dic_temp_F = {}
        for task in Task_list:
            dic_temp_P.setdefault(task, [])
            dic_temp_R.setdefault(task, [])
            dic_temp_F.setdefault(task, [])
            for each_dic in dic_valid_PRF_list:
                dic_temp_P[task].append(float(each_dic[task][0]))
                dic_temp_R[task].append(float(each_dic[task][1]))
                dic_temp_F[task].append(float(each_dic[task][2]))

        with open(file, "a") as f:
            Model_name = str(sys.argv[1:]) + "\n"
            f.writelines(Model_name)

            for task in Task_list:
                total_result_macro = "result |  P %3.3f (%3.3f), R %3.3f (%3.3f), F %3.3f (%3.3f) \n" % (
                np.average(dic_temp_P[task]), np.std(dic_temp_P[task]),
                np.average(dic_temp_R[task]), np.std(dic_temp_R[task]),
                np.average(dic_temp_F[task]), np.std(dic_temp_F[task]))
                f.writelines(task + ": " + total_result_macro)
            f.writelines("\n")


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
        if "entity_span" in task_list:
            f.write(str(dic_TP_FN_FP["entity_span"]) + "\n")

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
                                                                                  str(dic_TP_FN_FP[need_task][
                                                                                          sub_task])))
                        except KeyError:
                            pass
                    f.write("\n")
                f.write("\n")

                f.write("corpus level PRF (micro): \n")
                for task, PRF in dic_corpus_task_micro_P_R_F[corpus].items():
                    if task != "entity_span":
                        f.write("\t" + task + ": " + str([round(i, 3) for i in PRF]) + "\n")
                f.write("============================================================")


def get_sent_len(one_sent_tags, entity_BIOES_TAGS_filed):
    sentence_length = 0
    for tags in one_sent_tags:
        if int(tags) != entity_BIOES_TAGS_filed.vocab["[PAD]"]:
            sentence_length += 1
    return sentence_length


def get_entity_res_segment(one_batch_res, TAGS_filed, need_sep_flag):
    batch_segment_res = []
    for sent_index, one_sent_tags in enumerate(one_batch_res):
        sentence_length = get_sent_len(one_sent_tags, TAGS_filed)
        sentence = one_sent_tags[:sentence_length]
        sentence = [TAGS_filed.vocab.itos[int(i)] for i in sentence.cpu().numpy().tolist()]
        if need_sep_flag:
            sentence = get_triple_O_seg(sentence)
            sentence = formatted_outpus(sentence)
        batch_segment_res.append(sentence)
    return batch_segment_res


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


class EMA():
    def __init__(self, model, decay, device):
        self.device = device
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                try:
                    assert name in self.shadow
                except:
                    print(name)
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
