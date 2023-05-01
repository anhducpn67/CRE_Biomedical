#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import pickle
import warnings
import sys
import shutil
import argparse
import os
import copy
import random
import numpy as np
import torchtext
from sklearn.cluster import KMeans
from torchtext.legacy.data import Batch
import torch
import transformers
import torch.optim as optim

from utils import print_execute_time, Logger, record_detail_performance
from metric import report_performance
from my_modules import MyRelationClassifier, MyBertEncoder, MyModel
from data_loader import prepared_NER_data, prepared_RC_data, get_corpus_file_dic, make_model_data

parser = argparse.ArgumentParser(description="Bert Model")
parser.add_argument('--ID', default=0, type=int, help="Model's ID")
parser.add_argument('--BERT_MODEL', default="base", type=str, help="base, large")
parser.add_argument('--GPU', default="0", type=str)
parser.add_argument('--All_data', action='store_true', default=False)
parser.add_argument('--BATCH_SIZE', default=8, type=int)

parser.add_argument('--Average_Time', default=1, type=int)
parser.add_argument('--EPOCH', default=3, type=int)
parser.add_argument('--Min_train_performance_Report', default=1, type=int)
parser.add_argument('--EARLY_STOP_NUM', default=5, type=int)
parser.add_argument('--MEMORY_SIZE', default=100, type=int)

parser.add_argument('--Task_list', default=["entity_span", "entity_type", "relation"], nargs='+',
                    help="\"entity_span\", \"entity_type\", \"relation\"")

parser.add_argument('--Corpus_list', default=["DDI", "CPR", "Twi_ADE", "ADE", "PPI"], nargs='+',
                    help="\"DDI\", \"Twi_ADE\", \"ADE\", \"CPR\", \"PPI\"")
parser.add_argument('--Test_flag', action='store_true', default=False)
parser.add_argument('--Test_model_file', type=str, default="../result/save_model/???")

parser.add_argument('--Entity_Prep_Way', default="standard", type=str,
                    help="\"standard\" or \"entity_type_marker\"")

parser.add_argument('--LR_bert', default=1e-5, type=float)
parser.add_argument('--LR_classifier', default=1e-4, type=float)
parser.add_argument('--L2', default=1e-2, type=float)

parser.add_argument('--Weight_Loss', action='store_true', default=True)
parser.add_argument('--Loss', type=str, default="BCE", help="\"BCE\", \"CE\"")
parser.add_argument('--Min_weight', default=0.5, type=float)
parser.add_argument('--Max_weight', default=5, type=float)


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
warnings.filterwarnings("ignore")

file_model_save = "../result/save_model/" + f"Model_{args.ID}"
file_memory_save = "../result/save_memorized_samples/" + f"memory_{args.ID}.pkl"
file_training_performance = f'../result/detail_training/training_performance_{args.ID}.txt'

sys.stdout = Logger(filename=file_training_performance)

if args.BERT_MODEL == "large":
    args.model_path = "../../../Data/embedding/biobert_large"
    args.Word_embedding_size = 1024
    args.Hidden_Size_Common_Encoder = args.Word_embedding_size
elif args.BERT_MODEL == "base":
    args.model_path = "dmis-lab/biobert-base-cased-v1.1"
    args.Word_embedding_size = 768
    args.Hidden_Size_Common_Encoder = args.Word_embedding_size

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
device = torch.device("cpu")
OPTIMIZER = optim.AdamW
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


class TrainValidTest:
    def __init__(self, data_ID_2_corpus_dic, my_model,
                 train_set_list, valid_set_list, test_set_list,
                 sep_corpus_file_dic):

        self.my_model = my_model.to(device)
        self.model_state_dic = {}

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

        self.sep_corpus_file_dic = sep_corpus_file_dic

        self.memorized_samples = {}
        self.total_memorized_samples = []

        self.optimizer_bert_RC = OPTIMIZER(
            params=filter(lambda p: p.requires_grad, self.my_model.encoder.parameters()),
            lr=args.LR_bert, weight_decay=args.L2)

        self.optimizer_classifier = OPTIMIZER(
            params=filter(lambda p: p.requires_grad, self.my_model.my_relation_classifier.parameters()),
            lr=args.LR_classifier, weight_decay=args.L2)

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
        self.model_state_dic['epoch'] = epoch
        self.model_state_dic['my_model'] = self.my_model.state_dict()
        torch.save(self.model_state_dic, file_model_save)

    def one_epoch(self, corpus_name_list, iterator_dic, valid_test_flag):
        dic_batches_res = {"ID_list": [], "corpus_name_list": [], "relation": []}
        epoch_loss = 0
        count = 0

        temp_my_iterator_list = [[ner, rc] for ner, rc in zip(iterator_dic[0], iterator_dic[1])]

        if valid_test_flag == "train":
            total_examples = 0
            for batch_list in temp_my_iterator_list:
                total_examples += len(batch_list[0])
            print(f"Corpus {corpus_name_list}, Total examples {total_examples}")

        for batch_list in temp_my_iterator_list:
            count += 1

            # D_replay
            if valid_test_flag == "train" and corpus_name_list[0] != args.Corpus_list[0]:
                replay_batch_list = self.get_batch_memory()
                with torch.cuda.amp.autocast():
                    dic_loss_one_batch, _ = self.my_model.forward(replay_batch_list)

                batch_relation_loss = dic_loss_one_batch["relation"]
                batch_loss = 0.3 * batch_relation_loss

                batch_loss.backward()

                self.optimizer_bert_RC.step()
                self.optimizer_bert_RC.zero_grad()

                self.optimizer_classifier.step()
                self.optimizer_classifier.zero_grad()

            # D_train
            with torch.cuda.amp.autocast():
                dic_loss_one_batch, dic_res_one_batch = self.my_model.forward(batch_list)

            batch_relation_loss = dic_loss_one_batch["relation"]
            epoch_loss += batch_relation_loss

            if valid_test_flag == "train":
                batch_loss = 0.3 * batch_relation_loss

                batch_loss.backward()

                self.optimizer_bert_RC.step()
                self.optimizer_bert_RC.zero_grad()

                self.optimizer_classifier.step()
                self.optimizer_classifier.zero_grad()

            dic_batches_res["ID_list"].append(batch_list[0].ID)
            dic_batches_res["corpus_name_list"].append(corpus_name_list)
            dic_batches_res["relation"].append(dic_res_one_batch["relation"])

        dic_loss = {"relation": epoch_loss / count, "average": epoch_loss / count}

        return dic_loss, dic_batches_res

    def one_epoch_train(self, corpus_name_list):
        self.my_model.train()
        dic_loss, dic_batches_res = self.one_epoch(corpus_name_list, self.train_iterator_dic, "train")
        return dic_loss, dic_batches_res

    def one_epoch_valid(self, corpus_name_list):
        with torch.no_grad():
            self.my_model.eval()
            dic_loss, dic_batches_res = self.one_epoch(corpus_name_list, self.valid_iterator_dic, "valid")
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

    def get_batch_memory(self):
        batch_ids = random.sample(self.total_memorized_samples, k=args.BATCH_SIZE)
        NER_examples = []
        RC_examples = []
        for corpus_name, _ in self.memorized_samples.items():
            for NER_example, RC_example in zip(self.train_corpus_to_examples_dic[corpus_name][0],
                                               self.train_corpus_to_examples_dic[corpus_name][1]):
                if int(NER_example.ID) in batch_ids:
                    NER_examples.append(NER_example)
                    RC_examples.append(RC_example)

        batch_NER = Batch(data=NER_examples, dataset=self.train_set_list[0], device=device)
        batch_RE = Batch(data=RC_examples, dataset=self.train_set_list[1], device=device)
        return batch_NER, batch_RE

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
                dic_train_loss, dic_batches_train_res = self.one_epoch_train([corpus_name])
                if epoch >= args.Min_train_performance_Report:
                    report_performance(corpus_name, epoch, dic_train_loss,
                                       dic_batches_train_res,
                                       self.my_model.classifiers_dic,
                                       self.sep_corpus_file_dic,
                                       "train")

                    # Validating for each previous corpus
                    for i in range(0, idx_corpus + 1):
                        corpus_name_valid = corpus_list[i]
                        self.set_iterator_for_specific_corpus([corpus_name_valid])
                        dic_valid_loss, dic_batches_valid_res = self.one_epoch_valid(corpus_name_valid)
                        dic_valid_PRF, dic_valid_total_sub_task_P_R_F, dic_valid_corpus_task_micro_P_R_F, dic_valid_TP_FN_FP \
                            = report_performance(corpus_name_valid, epoch, dic_valid_loss,
                                                 dic_batches_valid_res,
                                                 self.my_model.classifiers_dic,
                                                 self.sep_corpus_file_dic,
                                                 "valid")

                    if dic_valid_PRF[args.Task_list[-1]][2] >= maxF:
                        early_stop_num = args.EARLY_STOP_NUM
                        maxF = dic_valid_PRF[args.Task_list[-1]][2]
                        record_best_dic = dic_valid_PRF
                        save_epoch = epoch
                        self.save_model(save_epoch)
                        file_detail_performance = f'../result/detail_performance/continual_{str(args.ID)}/performance_{str(corpus_name)}.txt'
                        os.makedirs(os.path.dirname(file_detail_performance), exist_ok=True)
                        record_detail_performance(epoch, dic_valid_total_sub_task_P_R_F, dic_valid_PRF,
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
            # shutil.copy(file_model_save, file_model_save + "_" + corpus_name)
            # ======================== Create memorized samples for current task ========================
            print(f"==================== Create memorized samples for {corpus_name} ====================")
            self.set_iterator_for_specific_corpus([corpus_name])
            temp_my_iterator_list = [[ner, rc] for ner, rc in
                                     zip(self.train_iterator_dic[0], self.train_iterator_dic[1])]
            all_embedding_representations = []
            for batch_NER, batch_RC in temp_my_iterator_list:
                # Step 1
                dic_res_one_batch = {}
                batch_gold_and_pred_entity_type_res = self.my_model.entity_type_extraction(batch_NER)
                dic_res_one_batch["entity_type"] = batch_gold_and_pred_entity_type_res
                batch_entity, batch_entity_type = self.my_model.get_relation_data(batch_RC, dic_res_one_batch)

                # Step 2
                batch_RE_gold_res_list = []
                for sent_index in range(len(batch_RC)):
                    gold_one_sent_all_sub_task_res_dic = []
                    for sub_task in self.my_model.my_relation_classifier.my_relation_sub_task_list:
                        for entity_pair in getattr(batch_RC, sub_task)[sent_index]:
                            entity_pair_span = \
                                self.my_model.my_relation_classifier.TAGS_Types_fields_dic[sub_task][1].vocab.itos[
                                    entity_pair]
                            if entity_pair_span != "[PAD]":
                                temp_pair = sorted(eval(entity_pair_span))
                                if temp_pair not in gold_one_sent_all_sub_task_res_dic:
                                    gold_one_sent_all_sub_task_res_dic.append(temp_pair)

                    batch_RE_gold_res_list.append(gold_one_sent_all_sub_task_res_dic)

                # Step 3
                with torch.no_grad():
                    batch_added_marker_entity_span_vec, batch_entity_pair_span_list, batch_sent_len_list = \
                        self.my_model.my_relation_classifier.memory_get_entity_pair_vec(batch_entity=batch_entity,
                                                                                        batch_tokens=batch_RC.tokens,
                                                                                        batch_entity_type=batch_entity_type,
                                                                                        batch_gold_RE=batch_RE_gold_res_list,
                                                                                        bert_RC=self.my_model.encoder)
                # Step 4
                for sent_index in range(len(batch_RC)):
                    ID_example = int(batch_RC.ID[sent_index])
                    for e_index in range(0, batch_sent_len_list[sent_index]):
                        embed = batch_added_marker_entity_span_vec[sent_index][e_index]
                        all_embedding_representations.append((embed, ID_example))

            self.memorized_samples[corpus_name] = select_data(all_embedding_representations)
            self.total_memorized_samples += self.memorized_samples[corpus_name]
            print(f"Number representation: ", len(all_embedding_representations))
            print(f"Number examples in memorized samples {corpus_name}: ", len(self.memorized_samples[corpus_name]))
            pickle.dump(self.memorized_samples, open(file_memory_save, "wb"))
            # ======================== Testing ========================
            self.test_fn(idx_corpus, file_model_save)
        return record_best_dic

    def test_fn(self, idx_corpus, file_model_save_path):
        print("==================== Testing ====================")
        print(file_model_save_path)
        print("Loading Model...")
        checkpoint = torch.load(file_model_save_path)
        self.my_model.load_state_dict(checkpoint['my_model'])
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
                dic_batches_res = {"ID_list": [], "tokens_list": [], "relation": []}
                epoch_loss = 0
                count = 0

                temp_my_iterator_list = [[ner, rc] for ner, rc in
                                         zip(self.test_iterator_dic[0], self.test_iterator_dic[1])]

                for batch_list in temp_my_iterator_list:
                    count += 1
                    with torch.cuda.amp.autocast():
                        dic_loss_one_batch, dic_res_one_batch = self.my_model.forward(batch_list)

                    batch_relation_loss = dic_loss_one_batch["relation"]
                    epoch_loss += batch_relation_loss

                    dic_batches_res["ID_list"].append(batch_list[0].ID)
                    dic_batches_res["tokens_list"].append(batch_list[0].tokens)
                    dic_batches_res["relation"].append(dic_res_one_batch["relation"])

                dic_loss = {"average": epoch_loss / count, "relation": epoch_loss / count}

            dic_test_PRF, dic_total_sub_task_P_R_F, dic_corpus_task_micro_P_R_F, dic_TP_FN_FP \
                = report_performance(corpus_name, 0, dic_loss, dic_batches_res,
                                     self.my_model.classifiers_dic,
                                     self.sep_corpus_file_dic,
                                     "train")

            file_detail_performance = f'../result/detail_performance/continual_{str(args.ID)}/{idx_corpus}/performance_{str(corpus_name)}.txt'
            os.makedirs(os.path.dirname(file_detail_performance), exist_ok=True)
            record_detail_performance(0, dic_total_sub_task_P_R_F, dic_test_PRF,
                                      file_detail_performance.replace('.txt', "_TAC.txt"),
                                      dic_corpus_task_micro_P_R_F, dic_TP_FN_FP,
                                      self.sep_corpus_file_dic, ["relation"], corpus_name, args.Average_Time)


@print_execute_time
def get_valid_performance(model_path):
    corpus_file_dic, sep_corpus_file_dic, pick_corpus_file_dic, combining_data_files_list, entity_type_list, relation_list \
        = get_corpus_file_dic(args.All_data, args.Corpus_list, args.Task_list, args.BERT_MODEL)

    make_model_data(args.BERT_MODEL, pick_corpus_file_dic, combining_data_files_list, entity_type_list, relation_list,
                    args.All_data)
    data_ID_2_corpus_dic = {"11111": "CPR", "22222": "DDI", "33333": "Twi_ADE", "44444": "ADE",
                            "55555": "PPI", "66666": "BioInfer", "77777": "Combine_ADE"}

    bert = transformers.BertModel.from_pretrained(model_path, is_decoder=False, add_cross_attention=False)
    tokenizer = transformers.BertTokenizer.from_pretrained(model_path)

    entity_type_list = ["Drug", "Gene", "Disease"]
    ADDITIONAL_SPECIAL_TOKENS_start = ["[Entity_only_entity_type_" + i + "]" for i in entity_type_list]
    ADDITIONAL_SPECIAL_TOKENS_end = ["[/Entity_only_entity_type_" + i + "]" for i in entity_type_list]
    ADDITIONAL_SPECIAL_TOKENS = ADDITIONAL_SPECIAL_TOKENS_start + ADDITIONAL_SPECIAL_TOKENS_end

    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    bert.resize_token_embeddings(len(tokenizer))

    my_bert_encoder = MyBertEncoder(bert, tokenizer, args, device)

    my_model = MyModel(my_bert_encoder, args, device)

    my_relation_classifier = MyRelationClassifier(args, tokenizer, device)
    classifiers_dic = dict(zip(["relation"], [my_relation_classifier]))

    corpus_name = list(corpus_file_dic.keys())[0]
    entity_type_num_list, relation_num_list, file_train_valid_test_list = corpus_file_dic[corpus_name]
    print("===============" + corpus_name + "===============")

    NER_train_set, NER_valid_set, NER_test_set, NER_TOKENS_fields, TAGS_Entity_Span_fields_dic, \
        TAGS_Entity_Type_fields_dic, TAGS_Entity_Span_And_Type_fields_dic, TAGS_sampled_entity_span_fields_dic, TAGS_sep_entity_fields_dic \
        = prepared_NER_data(tokenizer, file_train_valid_test_list, entity_type_num_list)

    RC_train_set, RC_valid_set, RC_test_set, RC_TOKENS_fields, TAGS_Relation_pair_fields_dic, TAGS_sampled_entity_span_fields_dic \
        = prepared_RC_data(tokenizer, file_train_valid_test_list, relation_num_list)
    
    train_set_list = [NER_train_set, RC_train_set]
    valid_set_list = [NER_valid_set, RC_valid_set]
    test_set_list = [NER_test_set, RC_test_set]

    my_relation_classifier.create_classifiers(TAGS_Relation_pair_fields_dic, TAGS_sampled_entity_span_fields_dic,
                                              TAGS_Entity_Type_fields_dic)

    my_model.add_classifiers(classifiers_dic)

    my_train_valid_test = TrainValidTest(data_ID_2_corpus_dic, my_model,
                                         train_set_list, valid_set_list, test_set_list,
                                         sep_corpus_file_dic)
    Average_Time_list = []
    for i in range(args.Average_Time):
        print("==========================" + str(i) + "=================================================")
        dic_res_PRF = my_train_valid_test.train_valid_fn()
        Average_Time_list.append(dic_res_PRF)


if __name__ == "__main__":
    print("GPU:", args.GPU)
    print("Bert:", args.BERT_MODEL)
    print("Batch size: ", args.BATCH_SIZE)
    print("Memory size: ", args.MEMORY_SIZE)
    print("LR_bert: ", args.LR_bert)
    print("LR_classifier: ", args.LR_classifier)
    print("All_data:", args.All_data)
    print("Corpus_list:", args.Corpus_list)
    print("Task_list:", args.Task_list)
    print("Entity_Prep_Way:", args.Entity_Prep_Way)
    print("Loss:", args.Loss)
    print("EARLY_STOP_NUM:", args.EARLY_STOP_NUM)
    print("Test_flag:", args.Test_flag)

    get_valid_performance(args.model_path)
