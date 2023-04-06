#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import pickle
import warnings
import sys
import shutil
import argparse
import os
import math
import copy
import random
import numpy as np
import torchtext
from sklearn.cluster import KMeans

from utils import print_execute_time, get_sent_len, Logger, record_each_performance, recored_detail_performance, \
    EMA, get_entity_type_rep_dic, log_gradient_updates, log_parameter_and_gradient_statistics
from metric import report_performance
from my_modules import My_Entity_Span_Classifier, My_Entity_Type_Classifier, My_Entity_Span_And_Type_Classifier, \
    My_Relation_Classifier, My_Bert_Encoder, My_Model
from data_loader import prepared_NER_data, prepared_RC_data, get_corpus_file_dic, make_model_data

parser = argparse.ArgumentParser(description="Bert Model")
parser.add_argument('--GPU', default="2", type=str)
parser.add_argument('--All_data', action='store_true', default=False)  # True False
parser.add_argument('--BATCH_SIZE', default=8, type=int)

parser.add_argument('--bert_model', default="base", type=str, help="base, large")
parser.add_argument('--Task_list', default=["entity_span", "entity_type", "relation"], nargs='+',
                    help=["entity_span", "entity_type", "entity_span_and_type", "relation"])
parser.add_argument('--Task_weights_dic', default="{'entity_span':0.0, 'entity_type':0.0, 'relation':0.3}", type=str)

parser.add_argument('--Corpus_list', default=["DDI", "CPR", "Twi_ADE", "ADE", "PPI"], nargs='+',
                    help=["DDI", "Twi_ADE", "ADE", "CPR", "PPI"])
parser.add_argument('--Random_ratio', default=1, type=float, help=">1 means mask all data from other corpus")
parser.add_argument('--Training_way', default="Continual_Training", type=str)
parser.add_argument('--Test_flag', action='store_true', default=False, help=[False, True])
parser.add_argument('--Test_Corpus', default=["TAC2019"], nargs='+',
                    help=["ADE", "Twi_ADE", "DDI", "CPR", "TAC2019"])  # TAC
parser.add_argument('--Test_model_file', type=str,
                    default="../result/save_model/???")

parser.add_argument('--Share_embedding', action='store_true', default=False, help=[False, True])
parser.add_argument('--Entity_Prep_Way', default="standard", type=str,
                    help=["standard", "entitiy_type_marker", "entity_type_embedding"])
parser.add_argument('--If_add_prototype', action='store_true', default=False)  # True False
parser.add_argument('--If_soft_share', action='store_true', default=False)  # True False
parser.add_argument('--Pick_lay_num', default=-1, type=int, help="-1 means last layer")

parser.add_argument('--Average_Time', default=1, type=int)
parser.add_argument('--EPOCH', default=30, type=int)
parser.add_argument('--Min_train_performance_Report', default=5, type=int)
parser.add_argument('--EARLY_STOP_NUM', default=5, type=int)
parser.add_argument('--MEMORY_SIZE', default=100, type=int)


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
parser.add_argument('--Type_emb_num', default=50, type=int)

parser.add_argument('--Relation_input', default="entity_span", type=str, help=["entity_span", "entity_span_and_type"])
parser.add_argument('--Only_relation', action='store_true', default=False)
parser.add_argument('--Num_warmup_epoch', default=3, type=int)
parser.add_argument('--Tensorboard', action='store_true', default=False)
parser.add_argument('--Optim', default="AdamW", type=str, help=["AdamW"])
parser.add_argument('--ID', default=0, type=int, help="Just for tensorboard reg")

parser.add_argument('--IF_CRF', action='store_true', default=False)
parser.add_argument('--Decay_epoch_num', default=40, type=float)
parser.add_argument('--Improve_Flag', action='store_true', default=False)
# AdamW  --LR_bert=5e-4 --LR_classfier=1e-3  70.285
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
warnings.filterwarnings("ignore")

file_param_record = '../result/param_record'
file_result = '../result/detail_results/result_' + str(sys.argv[1:]) + '.json'
file_model_save = "../result/save_model/" + "Model_" + str(sys.argv[1:])
file_training_performance = '../result/detail_training/training_' + str(sys.argv[1:]) + '.txt'

sys.stdout = Logger(filename=file_training_performance)
tensor_board_path = '../result/runs/' + str(sys.argv[1:])
try:
    shutil.rmtree(tensor_board_path)
except FileNotFoundError:
    pass

args.Task_weights_dic = eval(args.Task_weights_dic)
v_sum = 0
for k, v in args.Task_weights_dic.items():
    if k in args.Task_list:
        v_sum += v
# assert v_sum == 1

if args.bert_model == "large":
    args.model_path = "../../../Data/embedding/biobert_large"
    args.Word_embedding_size = 1024
    args.Hidden_Size_Common_Encoder = args.Word_embedding_size
elif args.bert_model == "base":
    args.model_path = "dmis-lab/biobert-base-cased-v1.1"
    args.Word_embedding_size = 768
    args.Hidden_Size_Common_Encoder = args.Word_embedding_size

from torch.utils.tensorboard import SummaryWriter
import torch
import transformers
import torch.optim as optim

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
OPTIMIZER = eval("optim." + args.Optim)
Embedding_requires_grad = True
BATCH_FIRST = True
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


# Use K-Means to select what samples to save, similar to at_least = 0
def select_data(all_embedding_representations):
    features = [embedding.detach().cpu().numpy() for embedding, ID in all_embedding_representations]

    num_clusters = min(args.MEMORY_SIZE, len(features))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

    mem_set = []
    for k in range(num_clusters):
        sel_index = np.argmin(distances[:, k])
        instance = all_embedding_representations[sel_index][1]
        if instance not in mem_set:
            mem_set.append(instance)
    return mem_set


class Train_valid_test:
    def __init__(self, data_ID_2_corpus_dic, my_model, tokenizer_list,
                 train_set_list, valid_set_list, test_set_list,
                 sep_corpus_file_dic, writer):

        self.my_model = my_model.to(device)
        self.tokenizer_list = tokenizer_list

        self.data_ID_2_corpus_dic = data_ID_2_corpus_dic

        self.train_set_list = train_set_list
        self.valid_set_list = valid_set_list
        self.test_set_list = test_set_list

        self.train_corpus_to_examples_dic = self.get_dic_data(train_set_list)
        self.valid_corpus_to_examples_dic = self.get_dic_data(valid_set_list)
        self.test_corpus_to_examples_dic = self.get_dic_data(test_set_list)

        self.train_iterator_dic = None
        self.valid_iterator_dic = None
        self.test_iterator_dic = None

        self.writer = writer
        # self.ema = ema

        self.sep_corpus_file_dic = sep_corpus_file_dic
        self.all_entity_type_classifier_list = ["only_entity_type_" + i for i in ['Gene', 'Drug', 'Disease']]
        self.all_entity_span_and_type_classifier_list = ["joint_entity_type_" + i for i in ['Gene', 'Drug', 'Disease']]
        self.all_relation_classifier_list = ["relation_" + i for i in
                                             ['Drug_Disease_interaction', 'Drug_Gene_interaction',
                                              'Drug_Drug_interaction', 'Gene_Gene_interaction']]

        self.memorized_samples = {}

        self.false_flag = False

        if "relation" in args.Task_list:
            self.optimizer_bert_RC = OPTIMIZER(
                params=filter(lambda p: p.requires_grad, self.my_model.bert_RC.parameters()), lr=args.LR_max_bert,
                weight_decay=args.L2)

        for task in ["relation"]:
            # set tasks optim
            my_parameters = getattr(self.my_model, "my_" + task + "_classifier").parameters()
            my_lr_min = getattr(args, "LR_min_" + str(task))
            my_lr_max = getattr(args, "LR_max_" + str(task))
            setattr(self, "optimizer_" + str(task), OPTIMIZER(params=filter(lambda p: p.requires_grad, my_parameters),
                                                              lr=my_lr_max, weight_decay=args.L2))
        self.entity_type_rep_dic = {}

    def sep_list(self, listTemp, n):
        step = math.ceil(len(listTemp) / n)
        return_list = []
        s_index = 0
        e_index = 0
        for i in range(n):
            e_index += step
            return_list.append(listTemp[s_index: e_index])
            s_index = e_index
        return return_list

    def get_dic_data(self, set_list):
        NER_data_dic = {}
        RC_data_dic = {}
        return_data_dict = {}

        corpus_list = copy.deepcopy(args.Corpus_list)

        NER_total_list = [example for example in set_list[0]]
        RC_total_list = [example for example in set_list[1]]

        for example in NER_total_list:
            corpus_name = self.data_ID_2_corpus_dic[str(int(example.ID))[:5]]
            NER_data_dic.setdefault(corpus_name, [])
            NER_data_dic[corpus_name].append(example)

        for example in RC_total_list:
            corpus_name = self.data_ID_2_corpus_dic[str(int(example.ID))[:5]]
            RC_data_dic.setdefault(corpus_name, [])
            RC_data_dic[corpus_name].append(example)

        for corpus_name in corpus_list:
            return_data_dict[corpus_name] = (NER_data_dic[corpus_name], RC_data_dic[corpus_name])

        return return_data_dict

    def save_model(self, epoch):
        self.model_state_dic = {}
        self.model_state_dic['epoch'] = epoch
        self.model_state_dic['entity_type_rep_dic'] = self.entity_type_rep_dic
        self.model_state_dic['my_model'] = self.my_model.state_dict()
        torch.save(self.model_state_dic, file_model_save)

    def one_epoch(self, corpus_name_list, iterator_dic, valid_test_flag, epoch):
        """
        dic_batches_res = {"entity_span":[  batch-num [ ( one_batch_pred_sub_res[batch_size[entity_num]], [batch.entity_span] ) ] ],
                            "entity_type":[], "relation":[]}
        """
        dic_batches_res = {"ID_list": [], "tokens_list": [], "corpus_name_list": [], "entity_span": [],
                           "entity_type": [], "entity_span_and_type": [], "relation": []}
        epoch_entity_span_loss = 0
        epoch_entity_type_loss = 0
        epoch_entity_span_and_type_loss = 0
        epoch_relation_loss = 0
        count = 0

        entity_type_no_need_list = []
        entity_span_and_type_no_need_list = []
        relation_no_need_list = []

        if valid_test_flag == "train":
            total_entity_type = []
            total_relation = []
            for corpus_name in corpus_name_list:
                total_entity_type += self.sep_corpus_file_dic[corpus_name]['entity_type']
                total_relation += self.sep_corpus_file_dic[corpus_name]['relation']

            if args.Random_ratio >= np.random.uniform(0, 1):
                self.false_flag = True
                if "entity_type" in args.Task_list:
                    for class_name in list(
                            set(self.all_entity_type_classifier_list).difference(set(total_entity_type))):
                        if hasattr(self.my_model.my_entity_type_classifier, "my_classifer_{0}".format(class_name)):
                            no_need_classifer = getattr(self.my_model.my_entity_type_classifier,
                                                        "my_classifer_{0}".format(class_name))
                            entity_type_no_need_list.append(no_need_classifer)
                            for p in no_need_classifer.parameters():
                                p.requires_grad = False
                if "relation" in args.Task_list:
                    for class_name in list(set(self.all_relation_classifier_list).difference(set(total_relation))):
                        if hasattr(self.my_model.my_relation_classifier, "my_classifer_{0}".format(class_name)):
                            no_need_classifer = getattr(self.my_model.my_relation_classifier,
                                                        "my_classifer_{0}".format(class_name))
                            relation_no_need_list.append(no_need_classifer)
                            for p in no_need_classifer.parameters():
                                p.requires_grad = False

        if "relation" in args.Task_list:
            temp_my_iterator_list = [[ner, rc] for ner, rc in zip(iterator_dic[0], iterator_dic[1])]
        else:
            raise Exception("error!")

        if valid_test_flag == "train":
            total_examples = 0
            for batch_list in temp_my_iterator_list:
                total_examples += len(batch_list[0])
            print(f"Corpus {corpus_name_list}, Total examples {total_examples}")

        for batch_list in temp_my_iterator_list:
            count += 1
            with torch.cuda.amp.autocast():
                dic_loss_one_batch, dic_res_one_batch = self.my_model.forward(batch_list, epoch,
                                                                              self.entity_type_rep_dic,
                                                                              valid_test_flag)

            batch_loss_list = []
            if "entity_span" in self.my_model.task_list:
                batch_entity_span_loss = dic_loss_one_batch["entity_span"]
                epoch_entity_span_loss += batch_entity_span_loss
                batch_loss_list.append(batch_entity_span_loss)
            if "entity_type" in self.my_model.task_list and dic_loss_one_batch["entity_type"] is not None:
                batch_entity_type_loss = dic_loss_one_batch["entity_type"]
                epoch_entity_type_loss += batch_entity_type_loss
                batch_loss_list.append(batch_entity_type_loss)
            if "entity_span_and_type" in self.my_model.task_list:
                batch_entity_span_and_type_loss = dic_loss_one_batch["entity_span_and_type"]
                epoch_entity_span_and_type_loss += batch_entity_span_and_type_loss
                batch_loss_list.append(batch_entity_span_and_type_loss)
            if "relation" in self.my_model.task_list:
                batch_relation_loss = dic_loss_one_batch["relation"]
                epoch_relation_loss += batch_relation_loss
                batch_loss_list.append(batch_relation_loss)

            if valid_test_flag == "train":

                batch_loss = torch.dot(
                    torch.tensor(list(args.Task_weights_dic.values()), device=device),
                    torch.stack(batch_loss_list))

                batch_loss.backward()

                if "relation" in args.Task_list:
                    self.optimizer_bert_RC.step()
                    self.optimizer_bert_RC.zero_grad()

                for task in ["relation"]:
                    getattr(self, "optimizer_" + str(task)).step()
                    getattr(self, "optimizer_" + str(task)).zero_grad()
                # self.ema.update()

            dic_batches_res["ID_list"].append(batch_list[0].ID)
            dic_batches_res["tokens_list"].append(batch_list[0].tokens)
            dic_batches_res["corpus_name_list"].append(corpus_name_list)
            for task in self.my_model.task_list:
                try:
                    dic_batches_res[task].append(dic_res_one_batch[task])
                except:
                    pass  # nothing wrong

        if valid_test_flag == "train" and self.false_flag:
            if "entity_type" in args.Task_list:
                for class_name in entity_type_no_need_list:
                    for p in class_name.parameters():
                        p.requires_grad = True
            if "entity_span_and_type" in args.Task_list:
                for class_name in entity_span_and_type_no_need_list:
                    for p in class_name.parameters():
                        p.requires_grad = True
            if "relation" in args.Task_list:
                for class_name in relation_no_need_list:
                    for p in class_name.parameters():
                        p.requires_grad = True

        epoch_loss = 0
        dic_loss = {"average": 0}
        if "entity_span" in self.my_model.task_list:
            dic_loss["entity_span"] = epoch_entity_span_loss / count
            epoch_loss += epoch_entity_span_loss / count
        if "entity_type" in self.my_model.task_list:
            dic_loss["entity_type"] = epoch_entity_type_loss / count
            epoch_loss += epoch_entity_type_loss / count
        if "entity_span_and_type" in self.my_model.task_list:
            dic_loss["entity_span_and_type"] = epoch_entity_span_and_type_loss / count
            epoch_loss += epoch_entity_span_and_type_loss / count
        if "relation" in self.my_model.task_list:
            dic_loss["relation"] = epoch_relation_loss / count
            epoch_loss += epoch_relation_loss / count

        dic_loss["average"] = epoch_loss / len(self.my_model.task_list)

        return dic_loss, dic_batches_res

    def one_epoch_train(self, corpus_name_list, epoch):
        self.my_model.train()
        dic_loss, dic_batches_res = self.one_epoch(corpus_name_list, self.train_iterator_dic, "train", epoch)
        return dic_loss, dic_batches_res

    def one_epoch_valid(self, corpus_name_list, epoch):
        with torch.no_grad():
            self.my_model.eval()
            # self.ema.apply_shadow()
            dic_loss, dic_batches_res = self.one_epoch(corpus_name_list, self.valid_iterator_dic, "valid", epoch)
            # self.ema.restore()
        # torch.cuda.empty_cache()
        return dic_loss, dic_batches_res

    def set_iterator_for_specific_corpus(self, corpus_name_list):
        NER_train, RC_train = self.train_set_list
        NER_valid, RC_valid = self.valid_set_list
        NER_test, RC_test = self.test_set_list
        NER_train.examples = []
        RC_train.examples = []
        NER_valid.examples = []
        RC_valid.examples = []
        NER_test.examples = []
        RC_test.examples = []

        for corpus_name in corpus_name_list:
            NER_train_corpus, RC_train_corpus = self.train_corpus_to_examples_dic[corpus_name]
            NER_valid_corpus, RC_valid_corpus = self.valid_corpus_to_examples_dic[corpus_name]
            NER_test_corpus, RC_test_corpus = self.test_corpus_to_examples_dic[corpus_name]
            NER_train.examples += NER_train_corpus
            RC_train.examples += RC_train_corpus
            NER_valid.examples += NER_valid_corpus
            RC_valid.examples += RC_valid_corpus
            NER_test.examples += NER_test_corpus
            RC_test.examples += RC_test_corpus

        NER_train_iterator, NER_valid_iterator, NER_test_iterator = torchtext.legacy.data.BucketIterator.splits(
            [NER_train, NER_valid, NER_test], batch_size=args.BATCH_SIZE, sort=False, shuffle=True,
            repeat=False, device=device)
        RC_train_iterator, RC_valid_iterator, RC_test_iterator = torchtext.legacy.data.BucketIterator.splits(
            [RC_train, RC_valid, RC_test], batch_size=args.BATCH_SIZE, sort=False, shuffle=True,
            repeat=False, device=device)
        self.train_iterator_dic = [NER_train_iterator, RC_train_iterator]
        self.valid_iterator_dic = [NER_valid_iterator, RC_valid_iterator]
        self.test_iterator_dic = [NER_test_iterator, RC_test_iterator]

    def set_iterator_for_memorized_samples(self):
        NER_examples = []
        RC_examples = []
        for corpus_name, ids in self.memorized_samples.items():
            for NER_example, RC_example in zip(self.train_corpus_to_examples_dic[corpus_name][0],
                                               self.train_corpus_to_examples_dic[corpus_name][1]):
                if int(NER_example.ID) in ids:
                    NER_examples.append(NER_example)
                    RC_examples.append(RC_example)

        NER_train, RC_train = self.train_set_list
        NER_valid, RC_valid = self.valid_set_list
        NER_test, RC_test = self.test_set_list
        NER_train.examples = NER_examples
        RC_train.examples = RC_examples
        NER_train_iterator, _, _ = torchtext.legacy.data.BucketIterator.splits(
            [NER_train, NER_valid, NER_test], batch_size=args.BATCH_SIZE, sort=False, shuffle=True,
            repeat=False, device=device)
        RC_train_iterator, _, _ = torchtext.legacy.data.BucketIterator.splits(
            [RC_train, RC_valid, RC_test], batch_size=args.BATCH_SIZE, sort=False, shuffle=True,
            repeat=False, device=device)
        self.train_iterator_dic = [NER_train_iterator, RC_train_iterator]

    @print_execute_time
    def train_valid_fn(self):
        corpus_list = copy.deepcopy(args.Corpus_list)
        print("start training...")
        for idx_corpus, corpus_name in enumerate(corpus_list):
            print('*' * 50)
            # ======================== Training for current corpus ========================
            print(f"==================== Training {corpus_name} ====================")
            maxF = 0
            save_epoch = 0
            early_stop_num = args.EARLY_STOP_NUM
            self.set_iterator_for_specific_corpus([corpus_name])
            for epoch in range(0, args.EPOCH + 1):
                dic_train_loss, dic_batches_train_res = self.one_epoch_train([corpus_name], epoch)
                if epoch >= args.Min_train_performance_Report:
                    report_performance(corpus_name, epoch, ["relation"], dic_train_loss,
                                       dic_batches_train_res,
                                       self.my_model.classifiers_dic,
                                       self.sep_corpus_file_dic,
                                       args.Improve_Flag, "train")

                    # Validating for each previous corpus
                    for i in range(0, idx_corpus + 1):
                        corpus_name_valid = corpus_list[i]
                        self.set_iterator_for_specific_corpus([corpus_name_valid])
                        dic_valid_loss, dic_batches_valid_res = self.one_epoch_valid(corpus_name_valid, 0)
                        dic_valid_PRF, dic_valid_total_sub_task_P_R_F, dic_valid_corpus_task_micro_P_R_F, dic_valid_TP_FN_FP \
                            = report_performance(corpus_name_valid, epoch, ["relation"], dic_valid_loss,
                                                 dic_batches_valid_res,
                                                 self.my_model.classifiers_dic,
                                                 self.sep_corpus_file_dic,
                                                 args.Improve_Flag, "valid")

                    if dic_valid_PRF[args.Task_list[-1]][2] >= maxF:
                        early_stop_num = args.EARLY_STOP_NUM
                        maxF = dic_valid_PRF[args.Task_list[-1]][2]
                        record_best_dic = dic_valid_PRF
                        save_epoch = epoch
                        self.save_model(save_epoch)
                        file_detail_performance = f'../result/detail_performance/continual_{str(args.ID)}/performance_{str(corpus_name)}.txt'
                        os.makedirs(os.path.dirname(file_detail_performance), exist_ok=True)
                        recored_detail_performance(epoch, dic_valid_total_sub_task_P_R_F, dic_valid_PRF,
                                                   file_detail_performance,
                                                   dic_valid_corpus_task_micro_P_R_F, dic_valid_TP_FN_FP,
                                                   self.sep_corpus_file_dic, ["relation"], corpus_name, args.Average_Time)
                    else:
                        early_stop_num -= 1

                    if early_stop_num <= 0:
                        print()
                        print("early stop, in epoch: %d !" % (int(save_epoch)))
                        for task in ["relation"]:
                            print(task, ": max F: %s, " % (str(record_best_dic[task][-1])))
                        break

                else:
                    print("epoch: ", epoch)
            print()
            print("Reach max epoch: %d !" % (int(save_epoch)))

            for task in ["relation"]:
                print(task, ": max F: %s, " % (str(record_best_dic[task][-1])))
            shutil.copy(file_model_save, file_model_save + "_" + corpus_name)
            # ======================== Create memorized samples for current task ========================
            print(f"==================== Create memorized samples for {corpus_name} ====================")
            self.set_iterator_for_specific_corpus([corpus_name])
            temp_my_iterator_list = [[ner, rc] for ner, rc in
                                     zip(self.train_iterator_dic[0], self.train_iterator_dic[1])]
            all_embedding_representations = []
            for batch_NER, batch_RC in temp_my_iterator_list:
                # Step 1
                batch_NER_gold_res_list = []
                for sent_index in range(len(batch_NER)):
                    gold_one_sent_all_sub_task_res_dic = {}
                    for sub_task in self.my_model.my_entity_type_classifier.my_entity_type_sub_task_list:
                        gold_one_sent_all_sub_task_res_dic.setdefault(sub_task, [])
                        for entity_pair in getattr(batch_NER, sub_task)[sent_index]:
                            entity_pair_span = \
                                self.my_model.my_entity_type_classifier.TAGS_Types_fileds_dic[sub_task][1].vocab.itos[
                                    entity_pair]
                            if str(entity_pair_span) != '[PAD]':
                                temp_pair = sorted(eval(entity_pair_span))
                                if temp_pair not in gold_one_sent_all_sub_task_res_dic[sub_task]:
                                    gold_one_sent_all_sub_task_res_dic[sub_task].append(temp_pair)
                    batch_NER_gold_res_list.append(gold_one_sent_all_sub_task_res_dic)
                # Step 2
                batch_RE_gold_res_list = []
                for sent_index in range(len(batch_RC)):
                    gold_one_sent_all_sub_task_res_dic = []
                    for sub_task in self.my_model.my_relation_classifier.my_relation_sub_task_list:
                        for entity_pair in getattr(batch_RC, sub_task)[sent_index]:
                            entity_pair_span = \
                                self.my_model.my_relation_classifier.TAGS_Types_fileds_dic[sub_task][1].vocab.itos[
                                    entity_pair]
                            if entity_pair_span != "[PAD]":
                                temp_pair = sorted(eval(entity_pair_span))
                                if temp_pair not in gold_one_sent_all_sub_task_res_dic:
                                    gold_one_sent_all_sub_task_res_dic.append(temp_pair)

                    batch_RE_gold_res_list.append(gold_one_sent_all_sub_task_res_dic)

                # Step 3
                batch_entity = []
                batch_entity_type = []
                batch_pred_entity_res = batch_RC.sampled_entity_span
                TAGS_filed = self.my_model.my_relation_classifier.TAGS_sampled_entity_span_fileds_dic["sampled_entity_span"][1]
                for sent_index, one_sent_entity in enumerate(batch_pred_entity_res):
                    one_sent_temp_list = one_sent_entity[:get_sent_len(one_sent_entity, TAGS_filed)]  # deal with PAD
                    one_sent_temp_list = [eval(TAGS_filed.vocab.itos[int(i)]) for i in
                                          one_sent_temp_list.cpu().numpy().tolist()]
                    batch_entity.append(sorted(one_sent_temp_list, key=lambda s: s[0]))

                    type_dic = batch_NER_gold_res_list[sent_index]
                    batch_entity_type.append(self.my_model.provide_dic_entity_type(one_sent_temp_list, type_dic))

                with torch.no_grad():
                    batch_added_marker_entity_span_vec, batch_entity_pair_span_list, batch_sent_len_list = \
                        self.my_model.my_relation_classifier.memory_get_entity_pair_span_vec(batch_entity=batch_entity,
                                                                                             batch_tokens=batch_RC.tokens,
                                                                                             batch_entity_type=batch_entity_type,
                                                                                             batch_gold_RE=batch_RE_gold_res_list,
                                                                                             bert_RC=self.my_model.bert_RC,
                                                                                             Entity_type_TAGS_Types_fileds_dic=self.my_model.my_entity_type_classifier.total_entity_type_to_index_dic)
                # Step 4
                for sent_index in range(len(batch_RC)):
                    ID_example = int(batch_RC.ID[sent_index])
                    for e_index in range(0, batch_sent_len_list[sent_index]):
                        embed = batch_added_marker_entity_span_vec[sent_index][e_index]
                        all_embedding_representations.append((embed, ID_example))

            self.memorized_samples[corpus_name] = select_data(all_embedding_representations)
            print(f"Number representation: ", len(all_embedding_representations))
            print(f"Number examples in memorized samples {corpus_name}: ", len(self.memorized_samples[corpus_name]))
            pickle.dump(self.memorized_samples, open("memorized_samples.pkl", "wb"))

            # ======================== Training with memorized_samples ========================
            if idx_corpus >= 1:  # Skip first task
                print("==================== Training with memorized_samples ====================")
                maxF = 0
                save_epoch = 0
                early_stop_num = args.EARLY_STOP_NUM
                previous_corpus = [corpus_list[i] for i in range(0, idx_corpus + 1)]
                for epoch in range(0, args.EPOCH + 1):
                    self.set_iterator_for_memorized_samples()
                    self.one_epoch_train(previous_corpus, epoch)
                    if epoch >= args.Min_train_performance_Report:
                        # Validating for all previous corpus
                        self.set_iterator_for_specific_corpus(previous_corpus)
                        dic_valid_loss, dic_batches_valid_res = self.one_epoch_valid(previous_corpus, 0)
                        dic_valid_PRF, dic_valid_total_sub_task_P_R_F, dic_valid_corpus_task_micro_P_R_F, dic_valid_TP_FN_FP \
                            = report_performance(previous_corpus, epoch, ["relation"], dic_valid_loss,
                                                 dic_batches_valid_res,
                                                 self.my_model.classifiers_dic,
                                                 self.sep_corpus_file_dic,
                                                 args.Improve_Flag, "valid")
                        if dic_valid_PRF[args.Task_list[-1]][2] >= maxF:
                            early_stop_num = args.EARLY_STOP_NUM
                            maxF = dic_valid_PRF[args.Task_list[-1]][2]
                            record_best_dic = dic_valid_PRF
                            save_epoch = epoch
                            self.save_model(save_epoch)
                            file_detail_performance = f'../result/detail_performance/continual_{str(args.ID)}/performance_memory_{str(previous_corpus)}.txt'
                            os.makedirs(os.path.dirname(file_detail_performance), exist_ok=True)
                            recored_detail_performance(epoch, dic_valid_total_sub_task_P_R_F, dic_valid_PRF,
                                                       file_detail_performance,
                                                       dic_valid_corpus_task_micro_P_R_F, dic_valid_TP_FN_FP,
                                                       self.sep_corpus_file_dic, ["relation"], previous_corpus,
                                                       args.Average_Time)
                        else:
                            early_stop_num -= 1

                        if early_stop_num <= 0:
                            print()
                            print("early stop, in epoch: %d !" % (int(save_epoch)))
                            for task in ["relation"]:
                                print(task, ": max F: %s, " % (str(record_best_dic[task][-1])))
                            break

            # ======================== Testing ========================
            self.test_fn(idx_corpus, file_model_save)
        return record_best_dic

    def test_fn(self, idx_corpus, file_model_save_path):
        print("==================== Testing ====================")
        print(file_model_save_path)
        print("Loading Model...")
        checkpoint = torch.load(file_model_save_path)
        self.my_model.load_state_dict(checkpoint['my_model'])
        self.entity_type_rep_dic = checkpoint['epoch']
        print("Loading success !")

        current_corpus_list = copy.deepcopy(args.Corpus_list)
        current_corpus_list = current_corpus_list[:(idx_corpus + 1)]

        corpus_list = []
        for corpus_name in current_corpus_list:
            corpus_list.append([corpus_name])
        corpus_list.append([corpus_name for corpus_name in current_corpus_list])

        for corpus_name in corpus_list:
            self.set_iterator_for_specific_corpus(corpus_name)
            with torch.no_grad():
                self.my_model.eval()
                dic_batches_res = {"ID_list": [], "tokens_list": [], "entity_span": [], "entity_type": [],
                                   "entity_span_and_type": [], "relation": []}
                epoch_entity_span_loss = 0
                epoch_entity_type_loss = 0
                epoch_entity_span_and_type_loss = 0
                epoch_relation_loss = 0
                count = 0

                if "relation" in args.Task_list:
                    temp_my_iterator_list = [[ner, rc] for ner, rc in zip(self.test_iterator_dic[0], self.test_iterator_dic[1])]

                for batch_list in temp_my_iterator_list:
                    count += 1
                    with torch.cuda.amp.autocast():
                        dic_loss_one_batch, dic_res_one_batch = self.my_model.forward(batch_list, 0,
                                                                                      self.entity_type_rep_dic,
                                                                                      "valid")
                    batch_loss_list = []
                    if "entity_span" in self.my_model.task_list:
                        batch_entity_span_loss = args.Task_weights_dic["entity_span"] * dic_loss_one_batch[
                            "entity_span"]
                        epoch_entity_span_loss += batch_entity_span_loss
                        batch_loss_list.append(batch_entity_span_loss)
                    if "entity_type" in self.my_model.task_list and dic_loss_one_batch["entity_type"] is not None:
                        batch_entity_type_loss = args.Task_weights_dic["entity_type"] * dic_loss_one_batch["entity_type"]
                        epoch_entity_type_loss += batch_entity_type_loss
                        batch_loss_list.append(batch_entity_type_loss)
                    if "entity_span_and_type" in self.my_model.task_list:
                        batch_entity_span_and_type_loss = args.Task_weights_dic["entity_span_and_type"] * \
                                                          dic_loss_one_batch["entity_span_and_type"]
                        epoch_entity_span_and_type_loss += batch_entity_span_and_type_loss
                        batch_loss_list.append(batch_entity_span_and_type_loss)
                    if "relation" in self.my_model.task_list:
                        batch_relation_loss = args.Task_weights_dic["relation"] * dic_loss_one_batch["relation"]
                        epoch_relation_loss += batch_relation_loss
                        batch_loss_list.append(batch_relation_loss)

                    dic_batches_res["ID_list"].append(batch_list[0].ID)
                    dic_batches_res["tokens_list"].append(batch_list[0].tokens)
                    for task in self.my_model.task_list:
                        try:
                            dic_batches_res[task].append(dic_res_one_batch[task])
                        except:
                            pass

                epoch_loss = 0
                dic_loss = {"average": 0}
                if "entity_span" in self.my_model.task_list:
                    dic_loss["entity_span"] = epoch_entity_span_loss / count
                    epoch_loss += epoch_entity_span_loss / count
                if "entity_type" in self.my_model.task_list:
                    dic_loss["entity_type"] = epoch_entity_type_loss / count
                    epoch_loss += epoch_entity_type_loss / count
                if "entity_span_and_type" in self.my_model.task_list:
                    dic_loss["entity_span_and_type"] = epoch_entity_span_and_type_loss / count
                    epoch_loss += epoch_entity_span_and_type_loss / count
                if "relation" in self.my_model.task_list:
                    dic_loss["relation"] = epoch_relation_loss / count
                    epoch_loss += epoch_relation_loss / count

                dic_loss["average"] = epoch_loss / len(self.my_model.task_list)

            dic_test_PRF, dic_total_sub_task_P_R_F, dic_corpus_task_micro_P_R_F, dic_TP_FN_FP \
                = report_performance(corpus_name, 0, ["relation"], dic_loss, dic_batches_res,
                                     self.my_model.classifiers_dic,
                                     self.sep_corpus_file_dic,
                                     args.Improve_Flag, "train")

            file_detail_performance = f'../result/detail_performance/continual_{str(args.ID)}/{idx_corpus}/performance_{str(corpus_name)}.txt'
            os.makedirs(os.path.dirname(file_detail_performance), exist_ok=True)
            recored_detail_performance(0, dic_total_sub_task_P_R_F, dic_test_PRF,
                                       file_detail_performance.replace('.txt', "_TAC.txt"),
                                       dic_corpus_task_micro_P_R_F, dic_TP_FN_FP,
                                       self.sep_corpus_file_dic, ["relation"], corpus_name, args.Average_Time)


@print_execute_time
def get_valid_performance(model_path):
    writer = SummaryWriter(tensor_board_path)
    corpus_file_dic, sep_corpus_file_dic, pick_corpus_file_dic, combining_data_files_list, entity_type_list, relation_list \
        = get_corpus_file_dic(args.All_data, args.Corpus_list, args.Task_list, args.bert_model)

    make_model_data(args.bert_model, pick_corpus_file_dic, combining_data_files_list, entity_type_list, relation_list,
                    args.All_data)
    data_ID_2_corpus_dic = {"11111": "CPR", "22222": "DDI", "33333": "Twi_ADE", "44444": "ADE",
                            "55555": "PPI", "66666": "BioInfer", "77777": "Combine_ADE"}
    bert_NER = transformers.BertModel.from_pretrained(model_path)
    tokenizer_NER = transformers.BertTokenizer.from_pretrained(model_path)

    bert_RC = transformers.BertModel.from_pretrained(model_path, is_decoder=args.If_soft_share,
                                                     add_cross_attention=args.If_soft_share)
    tokenizer_RC = transformers.BertTokenizer.from_pretrained(model_path)

    entitiy_type_list = ["Drug", "Gene", "Disease"]
    ADDITIONAL_SPECIAL_TOKENS_start = ["[Entity_only_entity_type_" + i + "]" for i in entitiy_type_list]
    ADDITIONAL_SPECIAL_TOKENS_end = ["[/Entity_only_entity_type_" + i + "]" for i in entitiy_type_list]
    ADDITIONAL_SPECIAL_TOKENS = ADDITIONAL_SPECIAL_TOKENS_start + ADDITIONAL_SPECIAL_TOKENS_end

    tokenizer_NER.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    bert_NER.resize_token_embeddings(len(tokenizer_NER))
    tokenizer_RC.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    bert_RC.resize_token_embeddings(len(tokenizer_RC))

    bert_list = []
    tokenizer_list = []
    bert_NER = My_Bert_Encoder(bert_NER, tokenizer_NER, args, device)
    bert_list.append(bert_NER)
    tokenizer_list.append(tokenizer_NER)

    bert_RC = My_Bert_Encoder(bert_RC, tokenizer_RC, args, device)
    bert_list.append(bert_RC)
    tokenizer_list.append(tokenizer_RC)

    my_model = My_Model(bert_list, args, device).to(device)

    my_entity_span_classifier = My_Entity_Span_Classifier(args, device)
    my_entity_type_classifier = My_Entity_Type_Classifier(args, device)
    my_entity_span_and_type_classifier = My_Entity_Span_And_Type_Classifier(args, device)
    my_relation_classifier = My_Relation_Classifier(args, tokenizer_RC, device)
    classifiers_dic = dict(zip(["entity_span", "entity_type", "entity_span_and_type", "relation"],
                               [my_entity_span_classifier, my_entity_type_classifier,
                                my_entity_span_and_type_classifier, my_relation_classifier]))

    train_set_list = []
    valid_set_list = []
    test_set_list = []

    # for index, (corpus_name, (entity_type_num_list, relation_num_list, file_train_valid_test_list)) in enumerate(corpus_file_dic.items()):
    corpus_name = list(corpus_file_dic.keys())[0]
    entity_type_num_list, relation_num_list, file_train_valid_test_list = corpus_file_dic[corpus_name]
    print("===============" + corpus_name + "===============")

    NER_train_set, NER_valid_set, NER_test_set, NER_TOKENS_fields, TAGS_Entity_Span_fields_dic, \
        TAGS_Entity_Type_fields_dic, TAGS_Entity_Span_And_Type_fields_dic, TAGS_sampled_entity_span_fields_dic, TAGS_sep_entity_fields_dic \
        = prepared_NER_data(args.BATCH_SIZE, device, tokenizer_NER, file_train_valid_test_list, entity_type_num_list)

    train_set_list.append(NER_train_set)
    valid_set_list.append(NER_valid_set)
    test_set_list.append(NER_test_set)

    my_entity_span_classifier.create_classifers(TAGS_Entity_Span_fields_dic)
    my_entity_type_classifier.create_classifers(TAGS_Entity_Type_fields_dic, TAGS_sep_entity_fields_dic)
    my_entity_span_and_type_classifier.create_classifers(TAGS_Entity_Span_And_Type_fields_dic)

    if "relation" in args.Task_list:
        RC_train_set, RC_valid_set, RC_test_set, RC_TOKENS_fields, TAGS_Relation_pair_fields_dic, TAGS_sampled_entity_span_fields_dic \
            = prepared_RC_data(args.BATCH_SIZE, device, tokenizer_RC, file_train_valid_test_list, relation_num_list)
        my_relation_classifier.create_classifers(TAGS_Relation_pair_fields_dic, TAGS_sampled_entity_span_fields_dic,
                                                 TAGS_Entity_Type_fields_dic)

        train_set_list.append(RC_train_set)
        valid_set_list.append(RC_valid_set)
        test_set_list.append(RC_test_set)

    my_model.add_classifers(classifiers_dic, args.Task_list)
    # ema = EMA(my_model, 0.999, device)
    # ema.register()

    my_train_valid_test = Train_valid_test(data_ID_2_corpus_dic, my_model, tokenizer_list,
                                           train_set_list, valid_set_list, test_set_list,
                                           sep_corpus_file_dic, writer)
    Average_Time_list = []
    for i in range(args.Average_Time):
        print("==========================" + str(i) + "=================================================")
        dic_res_PRF = my_train_valid_test.train_valid_fn()
        Average_Time_list.append(dic_res_PRF)  # increasing train cause wrong there !!

    record_each_performance(file_param_record, ["relation"], Average_Time_list)


if __name__ == "__main__":

    if args.Entity_Prep_Way != "standard":
        print("warning, Entity_Prep_Way is not default : ", args.Entity_Prep_Way)
        print()
    if args.If_add_prototype:
        print("warning, If_add_prototype is not default : ", args.If_add_prototype)
        print()
    if args.If_soft_share:
        print("warning, If_soft_share is not default : ", args.Entity_Prep_Way)
        print()

    print("GPU:", args.GPU)
    print("Bert:", args.bert_model)
    print("Batch size: ", args.BATCH_SIZE)
    print("Memory size: ", args.MEMORY_SIZE)
    print("LR_max_bert: ", args.LR_max_bert)
    print("LR_max_entity_span: ", args.LR_max_entity_span)
    print("LR_max_entity_type: ", args.LR_max_entity_type)
    print("LR_max_relation: ", args.LR_max_relation)
    print("All_data:", args.All_data)
    print("Corpus_list:", args.Corpus_list)
    print("Task_list:", args.Task_list)
    print("If_add_prototype:", args.If_add_prototype)
    print("Entity_Prep_Way:", args.Entity_Prep_Way)
    print("If_soft_share:", args.If_soft_share)
    print("Only_relation:", args.Only_relation)
    print("Task_weights_dic:", end="")
    for task in args.Task_list:
        print(args.Task_weights_dic[task], end=" ")
    print()
    print("Loss:", args.Loss)
    print("EARLY_STOP_NUM:", args.EARLY_STOP_NUM)
    print("Test_flag:", args.Test_flag)
    print("Training_way:", args.Training_way)

    get_valid_performance(args.model_path)
