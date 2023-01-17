#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import warnings
import sys
import shutil
import argparse
import os
from utils import print_execute_time, record_pred_str_res, Logger, record_each_performance, recored_detail_performance, EMA, get_entity_type_rep_dic, log_gradient_updates, log_parameter_and_gradient_statistics
from metric import report_performance
from my_modules import My_Entity_Span_Classifier, My_Entity_Type_Classifier, My_Entity_Span_And_Type_Classifier, My_Relation_Classifier, My_Bert_Encoder, My_Model
from data_loader import get_corpus_file_dic, prepared_data_1

parser = argparse.ArgumentParser(description="Bert Model")
parser.add_argument('--GPU', default="0", type=str)
parser.add_argument('--All_data', action='store_true', default=False)
parser.add_argument('--BATCH_SIZE', default=8, type=int)

parser.add_argument('--bert_model', default="base", type=str, help="base, large")
parser.add_argument('--Only_relation', action='store_true', default=False) # True False
parser.add_argument('--Only_entity_type', action='store_true', default=True) # True False
parser.add_argument('--Relation_input', default="entity_span_and_type", type=str, help=["entity_span", "entity_span_and_type"])

parser.add_argument('--Task_list', default=["entity_type"], nargs='+', help=["entity_span", "entity_type", "entity_span_and_type", "relation"])
parser.add_argument('--Task_weights_dic', default="{'entity_span':0.4, 'entity_type':1, 'entity_span_and_type':0.4, 'relation':1}", type=str)

parser.add_argument('--Corpus_list', default=["ADE"], nargs='+', help=["ADE", "Twi_ADE", "DDI", "CPR"])
parser.add_argument('--Train_way', default="Multi_Task_Training", type=str)

parser.add_argument('--Entity_Prep_Way', default="standard", type=str, help=["standard", "entitiy_type_marker", "entity_type_embedding"])
parser.add_argument('--If_add_prototype', action='store_true', default=False) # True False

parser.add_argument('--Average_Time', default=3, type=int)
parser.add_argument('--EPOCH', default=100, type=int)
parser.add_argument('--Min_train_performance_Report', default=0, type=int)
parser.add_argument('--EARLY_STOP_NUM', default=15, type=int)

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
parser.add_argument('--Loss', type=str, default="CE", help=["BCE", "CE", "FSL"])
parser.add_argument('--Min_weight', default=0.5, type=float)
parser.add_argument('--Max_weight', default=5, type=float)
parser.add_argument('--Tau', default=1.0, type=float)
parser.add_argument('--Type_emb_num', default=50, type=int)

parser.add_argument('--Num_warmup_epoch', default=3, type=int)
parser.add_argument('--Decay_epoch_num', default=40, type=float)
parser.add_argument('--Min_valid_num', default=0, type=int)
parser.add_argument('--Each_num_epoch_valid', default=1, type=int)
parser.add_argument('--STOP_threshold', default=0, type=float)
parser.add_argument('--Test_flag', action='store_true', default=False)
parser.add_argument('--Tensorboard', action='store_true', default=False)
parser.add_argument('--IF_CRF', action='store_true', default=False)
parser.add_argument('--Improve_Flag', action='store_true', default=False)
parser.add_argument('--Optim', default="AdamW", type=str, help=["AdamW"])
parser.add_argument('--ID', default=0, type=int, help="Just for tensorboard reg")
# AdamW  --LR_bert=5e-4 --LR_classfier=1e-3  70.285
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
warnings.filterwarnings("ignore")

file_param_record = '../result/param_record'
file_detail_performance = '../result/detail_performance/performance_' + str(sys.argv[1:]) + '.txt'
file_result = '../result/detail_results/result_' + str(sys.argv[1:]) + '.json'
file_model_save = "../result/save_model/" +"Model_"+ str(sys.argv[1:])
file_traning_performerance = '../result/detail_training/training_' + str(sys.argv[1:]) + '.txt'

sys.stdout = Logger(filename=file_traning_performerance)
tensor_board_path = '../result/runs/'+str(sys.argv[1:])
try: shutil.rmtree(tensor_board_path)
except FileNotFoundError: pass

args.Task_weights_dic=eval(args.Task_weights_dic)
v_sum = 0
for k,v in args.Task_weights_dic.items():
    if k in args.Task_list:
        v_sum+=v
assert v_sum==1

if args.Only_relation or args.Only_entity_type:
    assert len(args.Task_list)==1
else:
    assert len(args.Task_list)>1

# there is bug, can't use entity type:
assert args.Entity_Prep_Way=="standard"
assert args.If_add_prototype==False


if args.bert_model == "large":
    args.model_path = "../../../Data/embedding/biobert_large"
    args.Word_embedding_size = 1024
    args.Hidden_Size_Common_Encoder = args.Word_embedding_size
elif args.bert_model == "base":
    args.model_path = "../../../Data/embedding/biobert_base"
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


class Train_valid_test():
    def __init__(self, my_model, tokenizer, train_iterator, valid_iterator, sep_corpus_file_dic, ema, writer):
        self.my_model = my_model.to(device)
        self.current_epoch = 0
        self.tokenizer = tokenizer
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.writer = writer
        self.ema = ema
        num_warmup_step = args.Num_warmup_epoch * len(self.train_iterator)
        num_training_steps = args.Decay_epoch_num * len(self.train_iterator)

        self.sep_corpus_file_dic = sep_corpus_file_dic
        self.optimizer_bert_encoder = OPTIMIZER(params=filter(lambda p: p.requires_grad, self.my_model.my_bert_encoder.parameters()),
                                                lr=args.LR_max_bert, weight_decay=args.L2)

        # self.scheduler_bert_encoder = transformers.get_polynomial_decay_schedule_with_warmup(self.optimizer_bert_encoder, num_warmup_step, num_training_steps, lr_end=args.LR_max_bert, power=1.0)
        # self.scheduler_bert_encoder = OneCycleLR(self.optimizer_bert_encoder, max_lr=args.LR_max_bert, epochs=args.EPOCH+1, steps_per_epoch=len(self.train_iterator))
        self.scheduler_bert_encoder = transformers.get_constant_schedule_with_warmup(self.optimizer_bert_encoder, num_warmup_step)

        for task in self.my_model.task_list:
            # set tasks optim
            my_parameters = getattr(self.my_model, "my_"+task+"_classifier").parameters()
            my_lr_min = getattr(args, "LR_min_" + str(task))
            my_lr_max = getattr(args, "LR_max_" + str(task))
            setattr(self, "optimizer_"+str(task), OPTIMIZER(params=filter(lambda p: p.requires_grad, my_parameters),
                                                            lr=my_lr_max, weight_decay=args.L2))
            # set tasks scheduler
            my_optim = getattr(self, "optimizer_"+str(task))

            # setattr(self, "scheduler_" + str(task), transformers.get_polynomial_decay_schedule_with_warmup(my_optim, num_warmup_step, num_training_steps, lr_end=my_lr_min, power=1.0))
            # setattr(self, "scheduler_" + str(task), OneCycleLR(my_optim, max_lr=my_lr_max, epochs=args.EPOCH+1, steps_per_epoch=len(self.train_iterator), pct_start=0.1 ))
            setattr(self, "scheduler_" + str(task), transformers.get_constant_schedule_with_warmup(my_optim, num_warmup_step))


    def save_model(self, epoch):
        self.model_state_dic = {}
        self.model_state_dic['epoch'] = epoch
        self.model_state_dic['my_model'] = self.my_model.state_dict()
        self.model_state_dic['optimizer_bert_encoder'] = self.optimizer_bert_encoder.state_dict()
        for task in self.my_model.task_list:
            self.model_state_dic['optimizer_'+str(task)] = getattr(self, "optimizer_"+task).state_dict()
        torch.save(self.model_state_dic, file_model_save)

    def one_epoch(self, my_iterator, valid_test_flag, epoch, entity_type_rep_dic):
        """
        dic_batches_res = {"entity_span":[  batch-num [ ( one_batch_pred_sub_res[batch_size[entity_num]], [batch.entity_span] ) ] ],
                            "entity_type":[], "relation":[]}
        """

        dic_batches_res = {"ID_list":[], "tokens_list":[], "entity_span":[], "entity_type":[], "entity_span_and_type":[], "relation":[]}
        epoch_entity_span_loss = 0
        epoch_entity_type_loss = 0
        epoch_entity_span_and_type_loss = 0
        epoch_relation_loss = 0
        count = 0
        for batch in [batch for batch in my_iterator]:
            count+=1
            # with torch.cuda.amp.autocast():
            dic_loss_one_batch, dic_res_one_batch = self.my_model.forward(batch, epoch, entity_type_rep_dic, valid_test_flag)

            batch_loss_list = []
            if "entity_span" in self.my_model.task_list:
                batch_entity_span_loss = args.Task_weights_dic["entity_span"] * dic_loss_one_batch["entity_span"]
                epoch_entity_span_loss += batch_entity_span_loss
                batch_loss_list.append(batch_entity_span_loss)
            if "entity_type" in self.my_model.task_list and dic_loss_one_batch["entity_type"] != None:
                batch_entity_type_loss =  args.Task_weights_dic["entity_type"] * dic_loss_one_batch["entity_type"]
                epoch_entity_type_loss += batch_entity_type_loss
                batch_loss_list.append(batch_entity_type_loss)
            if "entity_span_and_type" in self.my_model.task_list:
                batch_entity_span_and_type_loss = args.Task_weights_dic["entity_span_and_type"] * dic_loss_one_batch["entity_span_and_type"]
                epoch_entity_span_and_type_loss += batch_entity_span_and_type_loss
                batch_loss_list.append(batch_entity_span_and_type_loss)
            if "relation" in self.my_model.task_list:
                batch_relation_loss = args.Task_weights_dic["relation"] * dic_loss_one_batch["relation"]
                epoch_relation_loss += batch_relation_loss
                batch_loss_list.append(batch_relation_loss)

            if valid_test_flag=="train":
                batch_loss = sum(batch_loss_list)
                batch_loss.backward()
                self.scheduler_bert_encoder.step()
                self.optimizer_bert_encoder.step()
                self.optimizer_bert_encoder.zero_grad()
                for task in self.my_model.task_list:
                    getattr(self, "scheduler_"+str(task)).step()
                    getattr(self, "optimizer_"+str(task)).step()
                    getattr(self, "optimizer_"+str(task)).zero_grad()
                self.ema.update()

            dic_batches_res["ID_list"].append(batch.ID)
            dic_batches_res["tokens_list"].append(batch.tokens)
            for task in self.my_model.task_list:
                try:
                    dic_batches_res[task].append(dic_res_one_batch[task])
                except:
                    pass

        epoch_loss = 0
        dic_loss = {"average":0 }
        if "entity_span" in self.my_model.task_list:
            dic_loss["entity_span"] = epoch_entity_span_loss/ count
            epoch_loss += epoch_entity_span_loss/ count
        if "entity_type" in self.my_model.task_list:
            dic_loss["entity_type"] = epoch_entity_type_loss/ count
            epoch_loss += epoch_entity_type_loss/ count
        if "entity_span_and_type" in self.my_model.task_list:
            dic_loss["entity_span_and_type"] = epoch_entity_span_and_type_loss/ count
            epoch_loss += epoch_entity_span_and_type_loss/ count
        if "relation" in self.my_model.task_list:
            dic_loss["relation"] = epoch_relation_loss/ count
            epoch_loss += epoch_relation_loss/ count

        dic_loss["average"] = epoch_loss / len(self.my_model.task_list)

        return dic_loss, dic_batches_res

    def one_epoch_train(self, epoch, entity_type_rep_dic):
        self.my_model.train()
        dic_loss, dic_batches_res= self.one_epoch(self.train_iterator, "train", epoch, entity_type_rep_dic)
        return dic_loss, dic_batches_res

    def one_epoch_valid(self, epoch, entity_type_rep_dic):
        with torch.no_grad():
            self.my_model.eval()
            self.ema.apply_shadow()
            dic_loss, dic_batches_res = self.one_epoch(self.valid_iterator, "valid", epoch, entity_type_rep_dic)
            self.ema.restore()
        # torch.cuda.empty_cache()
        return dic_loss, dic_batches_res

    def train_valid_fn(self, average_num):
        save_epoch = 0
        early_stop_num = args.EARLY_STOP_NUM
        maxF = 0

        print("start training...")
        for epoch in range(self.current_epoch, args.EPOCH+1):
            if args.If_add_prototype:
                entity_type_rep_dic = get_entity_type_rep_dic(self.train_iterator, self.valid_iterator, self.my_model, device)
            else:
                entity_type_rep_dic = {}

            dic_train_loss, dic_batches_train_res = self.one_epoch_train(epoch, entity_type_rep_dic)
            if epoch >= args.Min_train_performance_Report :
                dic_train_PRF, dic_total_sub_task_P_R_F, dic_corpus_task_micro_P_R_F, dic_TP_FN_FP \
                    = report_performance(epoch, self.my_model.task_list, dic_train_loss, dic_batches_train_res,
                                         self.my_model.classifiers_dic,
                                         self.sep_corpus_file_dic,
                                         args.Improve_Flag, "train")

                # for task in self.my_model.task_list:
                #     self.writer.add_scalars("Performance_P/"+task, {"train":dic_train_PRF[task][0]}, epoch)
                #     self.writer.add_scalars("Performance_R/"+task, {"train":dic_train_PRF[task][1]}, epoch)
                #     self.writer.add_scalars("Performance_F/"+task, {"train":dic_train_PRF[task][2]}, epoch)
                #     self.writer.add_scalars("Loss/"+task, {"train":dic_train_loss[task]}, epoch)
                #     self.writer.add_scalars("Learning rate/"+task, {task: getattr(self, "optimizer_"+str(task)).param_groups[0]["lr"]}, epoch)
                # self.writer.add_scalars("Learning rate/bert_encoder", {"bert_encoder": self.optimizer_bert_encoder.param_groups[0]["lr"]}, epoch)
                # log_gradient_updates(self.writer, self.my_model, epoch)
                # log_parameter_and_gradient_statistics(self.my_model, self.writer, epoch)
                # self.writer.close()

                if epoch % args.Each_num_epoch_valid == 0 and epoch >= args.Min_valid_num:
                    dic_valid_loss, dic_batches_valid_res = self.one_epoch_valid(epoch, entity_type_rep_dic)
                    dic_valid_PRF, dic_valid_total_sub_task_P_R_F, dic_valid_corpus_task_micro_P_R_F, dic_valid_TP_FN_FP \
                        = report_performance(epoch, self.my_model.task_list, dic_valid_loss, dic_batches_valid_res,
                                             self.my_model.classifiers_dic,
                                             self.sep_corpus_file_dic,
                                             args.Improve_Flag, "valid")

                    for task in self.my_model.task_list:
                        self.writer.add_scalars("Performance_P/"+task, {"valid":dic_valid_PRF[task][0]}, epoch)
                        self.writer.add_scalars("Performance_R/"+task, {"valid":dic_valid_PRF[task][1]}, epoch)
                        self.writer.add_scalars("Performance_F/"+task, {"valid":dic_valid_PRF[task][2]}, epoch)
                        self.writer.add_scalars("Loss/"+task, {"valid":dic_valid_loss[task]}, epoch)
                    self.writer.close()

                if epoch >= args.Min_valid_num:
                    if dic_valid_PRF[args.Task_list[-1]][2] >= maxF + args.STOP_threshold :
                        early_stop_num = args.EARLY_STOP_NUM
                        maxF = dic_valid_PRF[args.Task_list[-1]][2]
                        recored_best_dic = dic_valid_PRF
                        save_epoch = epoch
                        # self.save_model(save_epoch)

                        # record_pred_str_res(self.my_model.task_list, file_result, dic_batches_valid_res, average_num,
                        #                     self.tokenizer.convert_ids_to_tokens, self.my_model.classifiers_dic,
                        #                     args.IF_CRF)

                        recored_detail_performance(epoch, dic_valid_total_sub_task_P_R_F, dic_valid_PRF, file_detail_performance,
                                                   dic_valid_corpus_task_micro_P_R_F, dic_valid_TP_FN_FP,
                                                   self.sep_corpus_file_dic, args.Task_list, args.Corpus_list, args.Average_Time)
                    else :
                        early_stop_num -= args.Each_num_epoch_valid

                    if early_stop_num <= 0:
                        print()
                        print("early stop, in epoch: %d !" % (int(save_epoch)))
                        for task in args.Task_list:
                            print(task, ": max F: %s, " % (str(recored_best_dic[task][-1])))
                        return recored_best_dic
            else:
                print("epoch: ", epoch)
        print()
        print("Reach max epoch: %d !" % (int(save_epoch)))

        for task in args.Task_list:
            print(task, ": max F: %s, " % (str(recored_best_dic[task][-1])))
        # self.save_model(save_epoch)
        return recored_best_dic

@print_execute_time
def get_valid_performance(model_path):
    corpus_file_dic, sep_corpus_file_dic = get_corpus_file_dic(args.All_data, args.Corpus_list, args.Task_list)
    bert = transformers.BertModel.from_pretrained(model_path)
    tokenizer = transformers.BertTokenizer.from_pretrained(model_path)
    writer = SummaryWriter(tensor_board_path)

    # if args.Entity_Prep_Way == "entitiy_marker":
    #     ADDITIONAL_SPECIAL_TOKENS = ["<Entity>", "</Entity>"]
    #     tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    #     bert.resize_token_embeddings(len(tokenizer))
    # elif args.Entity_Prep_Way == "entitiy_type_marker" or "trained_entitiy_type_marker":
    #     entitiy_type_list = list(corpus_file_dic.values())[0][0]
    #     ADDITIONAL_SPECIAL_TOKENS_start = ["[Entity:"+i+"]" for i in entitiy_type_list]
    #     ADDITIONAL_SPECIAL_TOKENS_end = ["[/Entity:"+i+"]" for i in entitiy_type_list]
    #     ADDITIONAL_SPECIAL_TOKENS = ADDITIONAL_SPECIAL_TOKENS_start + ADDITIONAL_SPECIAL_TOKENS_end
    #     tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    #     bert.resize_token_embeddings(len(tokenizer))
    # dic_additional_token_id = {}
    # for i in ADDITIONAL_SPECIAL_TOKENS:
    #     dic_additional_token_id[i] = tokenizer.convert_tokens_to_ids(i)
    # args.dic_additional_token_id = dic_additional_token_id

    bert_encoder = My_Bert_Encoder(bert, tokenizer, args, device)
    my_model = My_Model(bert_encoder, args, device).to(device)
    my_entity_span_classifier = My_Entity_Span_Classifier(args, device)
    my_entity_type_classifier = My_Entity_Type_Classifier(args, device)
    my_entity_span_and_type_classifier = My_Entity_Span_And_Type_Classifier(args, device)
    my_relation_classifier = My_Relation_Classifier(args, device)
    classifiers_dic = dict(zip(["entity_span", "entity_type", "entity_span_and_type", "relation"],
                               [my_entity_span_classifier, my_entity_type_classifier, my_entity_span_and_type_classifier, my_relation_classifier]))

    Average_Time_list = []
    for i in range(args.Average_Time):
        print("==========================" + str(i) + "=================================================")
        for index, (corpus_name, (entity_type_num_list, relation_num_list, file_train_valid_test_list)) in enumerate(corpus_file_dic.items()):
            print("===============" + corpus_name + "===============")
            train_iterator, valid_iterator, test_iterator, TOEKNS_fileds, TAGS_Entity_Span_fileds_dic, \
            TAGS_Entity_Type_fileds_dic, TAGS_Entity_Span_And_Type_fileds_dic, TAGS_Relation_pair_fileds_dic, \
            TAGS_sampled_entity_span_fileds_dic, TAGS_sep_entity_fileds_dic = \
                prepared_data_1(args.BATCH_SIZE, device, tokenizer, file_train_valid_test_list, entity_type_num_list, relation_num_list)

            if index == 0:
                my_entity_span_classifier.create_classifers(TAGS_Entity_Span_fileds_dic)
            my_entity_type_classifier.create_classifers(TAGS_Entity_Type_fileds_dic, TAGS_sep_entity_fileds_dic)
            my_entity_span_and_type_classifier.create_classifers(TAGS_Entity_Span_And_Type_fileds_dic)
            my_relation_classifier.create_classifers(TAGS_Relation_pair_fileds_dic, TAGS_sampled_entity_span_fileds_dic, TAGS_Entity_Type_fileds_dic)

            my_model.add_classifers(classifiers_dic, args.Task_list)
            ema = EMA(my_model, 0.999, device)
            ema.register()

            my_train_valid_test = Train_valid_test(my_model, tokenizer, train_iterator, test_iterator, sep_corpus_file_dic, ema, writer)

            if args.Test_flag:
                dic_res_PRF = my_train_valid_test.test_fn(file_model_save)
            else:
                dic_res_PRF = my_train_valid_test.train_valid_fn(i)

            Average_Time_list.append(dic_res_PRF)  # inceasing train cause wrong there !!
    record_each_performance(file_param_record, args.Task_list, Average_Time_list)



if __name__ == "__main__":
    print("GPU:", args.GPU)
    print("All_data:", args.All_data)
    print("Corpus_list:", args.Corpus_list)
    print("Task_list:", args.Task_list)
    print("Task_weights_dic:", end="")
    for task in args.Task_list:
        print(args.Task_weights_dic[task], end=" ")
    print()
    print("Loss:", args.Loss)
    print("BATCH_SIZE:", args.BATCH_SIZE)
    print("EARLY_STOP_NUM:", args.EARLY_STOP_NUM)
    print("Weight_Loss:", args.Weight_Loss)
    if args.Weight_Loss:
        print("Min_weight:", args.Min_weight)
        print("Max_weight:", args.Max_weight)
    get_valid_performance(args.model_path)
