#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import pickle
import warnings
import sys
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
from my_modules import MyRelationClassifier, MyEncoder, MyModel
from data_loader import prepared_data, get_corpus_list_information, make_model_data

parser = argparse.ArgumentParser(description="Bert model")
parser.add_argument('--ID', default=0, type=int, help="model's ID")
parser.add_argument('--BERT_MODEL', default="base", type=str, help="base, large")
parser.add_argument('--GPU', default="0", type=str)
parser.add_argument('--ALL_DATA', action='store_true', default=False)
parser.add_argument('--BATCH_SIZE', default=8, type=int)

parser.add_argument('--EPOCH', default=3, type=int)
parser.add_argument('--MIN_EPOCH_VALID', default=1, type=int)
parser.add_argument('--EARLY_STOP_NUM', default=5, type=int)
parser.add_argument('--MEMORY_SIZE', default=100, type=int)

parser.add_argument('--Corpus_list', default=["Combine_ADE", "DDI", "CPR"], nargs='+',
                    help="\"DDI\", \"Twi_ADE\", \"ADE\", \"CPR\", \"PPI\"")
parser.add_argument('--Test_flag', action='store_true', default=False)
parser.add_argument('--Test_model_file', type=str, default="../result/save_model/???")

parser.add_argument('--Entity_Prep_Way', default="entity_type_marker", type=str,
                    help="\"standard\" or \"entity_type_marker\"")

parser.add_argument('--LR_bert', default=1e-5, type=float)
parser.add_argument('--LR_classifier', default=1e-5, type=float)
parser.add_argument('--L2', default=1e-2, type=float)

parser.add_argument('--Weight_Loss', action='store_true', default=True)
parser.add_argument('--Loss', type=str, default="BCE", help="\"BCE\", \"CE\"")
parser.add_argument('--Min_weight', default=0.5, type=float)
parser.add_argument('--Max_weight', default=5, type=float)


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
warnings.filterwarnings("ignore")

file_model_save = "../result/save_model/" + f"model_{args.ID}"
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
                 train_dataset, valid_dataset, test_dataset,
                 corpus_information, relation_list):

        self.my_model = my_model.to(device)
        self.model_state_dic = {}

        self.data_ID_2_corpus_dic = data_ID_2_corpus_dic

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        self.train_corpus_to_examples_dic = self.get_corpus_to_examples(train_dataset)
        self.valid_corpus_to_examples_dic = self.get_corpus_to_examples(valid_dataset)
        self.test_corpus_to_examples_dic = self.get_corpus_to_examples(test_dataset)

        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

        self.corpus_information = corpus_information
        self.relation_list = relation_list

        self.memorized_samples = {}
        self.total_memorized_samples = []

        self.optimizer_encoder = optim.AdamW(
            params=filter(lambda p: p.requires_grad, self.my_model.encoder.parameters()),
            lr=args.LR_bert, weight_decay=args.L2)

        self.optimizer_classifier = optim.AdamW(
            params=filter(lambda p: p.requires_grad, self.my_model.classifier.parameters()),
            lr=args.LR_classifier, weight_decay=args.L2)

    def get_corpus_to_examples(self, dataset):
        corpus_to_example = {}

        for example in dataset:
            corpus_name = self.data_ID_2_corpus_dic[str(int(example.ID))[:5]]
            corpus_to_example.setdefault(corpus_name, [])
            corpus_to_example[corpus_name].append(example)

        return corpus_to_example

    def save_model(self, epoch):
        self.model_state_dic['epoch'] = epoch
        self.model_state_dic['my_model'] = self.my_model.state_dict()
        torch.save(self.model_state_dic, file_model_save)

    def one_epoch(self, corpus_list, batch_iterator, valid_test_flag):
        dic_batches_res = {"ID": [], "relation": []}
        epoch_loss = 0
        count = 0

        if valid_test_flag == "train":
            total_examples = 0
            for batch in batch_iterator:
                total_examples += len(batch)
            print(f"Corpus {corpus_list}, Total examples {total_examples}")

        for batch in batch_iterator:
            count += 1

            # D_replay
            if valid_test_flag == "train" and corpus_list[0] != args.Corpus_list[0]:
                replay_batch = self.get_batch_memory()
                with torch.cuda.amp.autocast():
                    batch_loss, _ = self.my_model.forward(replay_batch)

                batch_loss = 0.3 * batch_loss

                batch_loss.backward()

                self.optimizer_encoder.step()
                self.optimizer_encoder.zero_grad()

                self.optimizer_classifier.step()
                self.optimizer_classifier.zero_grad()

            # D_train
            with torch.cuda.amp.autocast():
                batch_loss, batch_res = self.my_model.forward(batch)

            epoch_loss += batch_loss

            if valid_test_flag == "train":
                batch_loss = 0.3 * batch_loss

                batch_loss.backward()

                self.optimizer_encoder.step()
                self.optimizer_encoder.zero_grad()

                self.optimizer_classifier.step()
                self.optimizer_classifier.zero_grad()

            dic_batches_res["ID"].append(batch.ID)
            dic_batches_res["relation"].append(batch_res)

        dic_loss = {"relation": epoch_loss, "average": epoch_loss / count}

        return dic_loss, dic_batches_res

    def one_epoch_train(self, corpus_list):
        self.my_model.train()
        dic_loss, dic_batches_res = self.one_epoch(corpus_list, self.train_iterator, "train")
        return dic_loss, dic_batches_res

    def one_epoch_valid(self, corpus_list):
        with torch.no_grad():
            self.my_model.eval()
            dic_loss, dic_batches_res = self.one_epoch(corpus_list, self.valid_iterator, "valid")
        return dic_loss, dic_batches_res

    def set_iterator_for_corpus_list(self, corpus_list):
        self.train_dataset.examples = []
        self.valid_dataset.examples = []
        self.test_dataset.examples = []

        for corpus in corpus_list:
            self.train_dataset.examples += self.train_corpus_to_examples_dic[corpus]
            self.valid_dataset.examples += self.valid_corpus_to_examples_dic[corpus]
            self.test_dataset.examples += self.test_corpus_to_examples_dic[corpus]

        train_iterator, valid_iterator, test_iterator = torchtext.legacy.data.BucketIterator.splits(
            [self.train_dataset, self.valid_dataset, self.test_dataset],
            batch_size=args.BATCH_SIZE, sort=False, shuffle=True, repeat=False, device=device)
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.test_iterator = test_iterator

    def get_batch_memory(self):
        batch_ids = random.sample(self.total_memorized_samples, k=args.BATCH_SIZE)
        batch_examples = []
        for corpus_name, _ in self.memorized_samples.items():
            for example in self.train_corpus_to_examples_dic[corpus_name]:
                if int(example.ID) in batch_ids:
                    batch_examples.append(example)

        batch = Batch(data=batch_examples, dataset=self.train_dataset, device=device)
        return batch

    @print_execute_time
    def train_valid_fn(self):
        corpus_list = copy.deepcopy(args.Corpus_list)
        print("start training...")
        for idx_corpus, corpus_name in enumerate(corpus_list):
            print('*' * 50)
            print(f"==================== Training {corpus_name} ====================")
            maxF = 0
            save_epoch = 0
            early_stop_num = args.EARLY_STOP_NUM
            self.set_iterator_for_corpus_list([corpus_name])
            for epoch in range(0, args.EPOCH):
                dic_train_loss, dic_batches_train_res = self.one_epoch_train([corpus_name])
                if epoch >= args.MIN_EPOCH_VALID:
                    report_performance(corpus_name, epoch, dic_train_loss,
                                       dic_batches_train_res,
                                       self.relation_list,
                                       "train")

                    # Validating for each previous corpus
                    for i in range(0, idx_corpus + 1):
                        corpus_name_valid = corpus_list[i]
                        self.set_iterator_for_corpus_list([corpus_name_valid])
                        dic_valid_loss, dic_batches_valid_res = self.one_epoch_valid(corpus_name_valid)
                        micro_P_R_F1, relation_P_R_F1, relation_TP_FN_FP = report_performance(corpus_name_valid, epoch,
                                                                                              dic_valid_loss,
                                                                                              dic_batches_valid_res,
                                                                                              self.relation_list,
                                                                                              "valid")

                    if micro_P_R_F1[2] >= maxF:
                        early_stop_num = args.EARLY_STOP_NUM
                        maxF = micro_P_R_F1[2]
                        save_epoch = epoch
                        self.save_model(save_epoch)
                        file_detail_performance = f'../result/detail_performance/continual_{str(args.ID)}/performance_{corpus_name}.txt'
                        os.makedirs(os.path.dirname(file_detail_performance), exist_ok=True)
                        record_detail_performance(relation_P_R_F1, micro_P_R_F1, file_detail_performance,
                                                  relation_TP_FN_FP, self.corpus_information, [corpus_name])
                    else:
                        early_stop_num -= 1

                    if early_stop_num <= 0:
                        print("Early stop, in epoch: %d !" % (int(save_epoch)))
                        print("Max micro-F1: %s " % (str(maxF)))
                        break
                else:
                    print("Epoch: ", epoch)
            print()
            print("Reach max epoch: %d !" % (int(save_epoch)))
            print("Max micro-F1: %s " % (str(maxF)))
            # shutil.copy(file_model_save, file_model_save + "_" + corpus_name)
            print(f"==================== Create memorized samples for {corpus_name} ====================")
            self.set_iterator_for_corpus_list([corpus_name])
            all_embedding_representations = []
            for batch in self.train_iterator:
                # Step 1
                batch_entity, batch_entity_type = self.my_model.get_relation_data(batch)

                # Step 2
                batch_RE_gold_res_list = []
                for sent_index in range(len(batch)):
                    gold_one_sent_all_sub_task_res_dic = []
                    for relation in self.my_model.classifier.relation_list:
                        for entity_pair in getattr(batch, relation)[sent_index]:
                            entity_pair_span = self.my_model.classifier.TAGS_Types_fields_dic[relation][1].vocab.itos[entity_pair]
                            if entity_pair_span != "[PAD]":
                                temp_pair = sorted(eval(entity_pair_span))
                                if temp_pair not in gold_one_sent_all_sub_task_res_dic:
                                    gold_one_sent_all_sub_task_res_dic.append(temp_pair)

                    batch_RE_gold_res_list.append(gold_one_sent_all_sub_task_res_dic)

                # Step 3
                with torch.no_grad():
                    batch_added_marker_entity_span_vec, batch_entity_pair_span_list, batch_sent_len_list = \
                        self.my_model.encoder.memory_get_entity_pair_rep(batch_entity=batch_entity,
                                                                         batch_tokens=batch.tokens,
                                                                         batch_entity_type=batch_entity_type,
                                                                         batch_gold_RE=batch_RE_gold_res_list)
                # Step 4
                for sent_index in range(len(batch)):
                    ID_example = int(batch.ID[sent_index])
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

    def test_fn(self, idx_corpus, file_model_save_path):
        print("==================== Testing ====================")
        print(file_model_save_path)
        print("Loading model...")
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
            self.set_iterator_for_corpus_list(corpus_name)
            with torch.no_grad():
                self.my_model.eval()
                dic_batches_res = {"ID": [], "relation": []}
                epoch_loss = 0
                count = 0

                for batch in self.test_iterator:
                    count += 1
                    with torch.cuda.amp.autocast():
                        batch_loss, batch_res = self.my_model.forward(batch)

                    epoch_loss += batch_loss

                    dic_batches_res["ID"].append(batch.ID)
                    dic_batches_res["relation"].append(batch_res)

                dic_loss = {"average": epoch_loss / count, "relation": epoch_loss}

            micro_P_R_F1, relation_P_R_F1, relation_TP_FN_FP = report_performance(corpus_name, 0, dic_loss,
                                                                                  dic_batches_res,
                                                                                  self.relation_list,
                                                                                  "train")

            file_detail_performance = f'../result/detail_performance/continual_{str(args.ID)}/{idx_corpus}/performance_{str(corpus_name)}.txt'
            os.makedirs(os.path.dirname(file_detail_performance), exist_ok=True)
            record_detail_performance(relation_P_R_F1, micro_P_R_F1,
                                      file_detail_performance.replace('.txt', "_TAC.txt"),
                                      relation_TP_FN_FP, self.corpus_information, corpus_name)


@print_execute_time
def get_valid_performance(model_path):
    corpus_information, combining_data_files_list, entity_type_list, relation_list \
        = get_corpus_list_information(args.ALL_DATA, args.Corpus_list, args.BERT_MODEL)

    make_model_data(args.BERT_MODEL, corpus_information, combining_data_files_list, entity_type_list, relation_list,
                    args.ALL_DATA)
    data_ID_2_corpus_dic = {"11111": "CPR", "22222": "DDI", "33333": "Twi_ADE", "44444": "ADE",
                            "55555": "PPI", "66666": "BioInfer", "77777": "Combine_ADE"}

    bert = transformers.BertModel.from_pretrained(model_path, is_decoder=False, add_cross_attention=False)
    tokenizer = transformers.BertTokenizer.from_pretrained(model_path)

    ADDITIONAL_SPECIAL_TOKENS_start = ["[Entity_" + entity_type + "]" for entity_type in entity_type_list]
    ADDITIONAL_SPECIAL_TOKENS_end = ["[/Entity_" + entity_type + "]" for entity_type in entity_type_list]
    ADDITIONAL_SPECIAL_TOKENS = ADDITIONAL_SPECIAL_TOKENS_start + ADDITIONAL_SPECIAL_TOKENS_end

    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    bert.resize_token_embeddings(len(tokenizer))

    my_bert_encoder = MyEncoder(bert, tokenizer, args, device)

    my_relation_classifier = MyRelationClassifier(args, device)

    my_model = MyModel(my_bert_encoder, my_relation_classifier, args, device)

    train_dataset, valid_dataset, test_dataset, TAGS_Entity_Type_fields_dic, TAGS_Relation_fields_dic, TAGS_sep_entity_fields_dic \
        = prepared_data(tokenizer, combining_data_files_list, entity_type_list, relation_list)

    my_relation_classifier.create_classifiers(TAGS_Relation_fields_dic,
                                              TAGS_sep_entity_fields_dic,
                                              TAGS_Entity_Type_fields_dic)

    my_train_valid_test = TrainValidTest(data_ID_2_corpus_dic, my_model,
                                         train_dataset, valid_dataset, test_dataset,
                                         corpus_information, relation_list)

    my_train_valid_test.train_valid_fn()


if __name__ == "__main__":
    print("GPU:", args.GPU)
    print("Bert:", args.BERT_MODEL)
    print("Batch size: ", args.BATCH_SIZE)
    print("Memory size: ", args.MEMORY_SIZE)
    print("LR_bert: ", args.LR_bert)
    print("LR_classifier: ", args.LR_classifier)
    print("ALL_DATA:", args.ALL_DATA)
    print("Corpus_list:", args.Corpus_list)
    print("Entity_Prep_Way:", args.Entity_Prep_Way)
    print("Loss:", args.Loss)
    print("EARLY_STOP_NUM:", args.EARLY_STOP_NUM)
    print("Test_flag:", args.Test_flag)

    get_valid_performance(args.model_path)
