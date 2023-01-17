from torchcrf import CRF
import torch.nn as nn
import torch
import torchtext
import numpy as np
from utils import get_entity_res_segment, print_execute_time, get_sent_len
from itertools import combinations
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import copy

class My_Bert_Encoder(nn.Module):
    def __init__(self, bert, tokenizer, args, device):
        super(My_Bert_Encoder, self).__init__()
        self.to(device)
        self.args = args
        self.device = device
        self.bert = bert
        self.tokenizer = tokenizer

    def forward(self, batch_inputs, ignore_index=0, encoder_hidden_states=None):
        tokens_tensor = self.get_bert_input(batch_inputs)
        attention_mask = self.get_attention_mask(tokens_tensor, ignore_index)
        position_ids = self.get_position_ids(tokens_tensor)
        common_embedding = self.bert(tokens_tensor,  output_hidden_states=True, attention_mask=attention_mask,
                                     position_ids=position_ids, encoder_hidden_states=encoder_hidden_states)

        last_common_embedding = common_embedding[0]
        n_layer_common_embedding = common_embedding[2][self.args.Pick_lay_num]

        return last_common_embedding, n_layer_common_embedding

    def get_bert_input(self, batch_inputs):
        input_tensor = torch.tensor(batch_inputs, device=self.device)
        return input_tensor

    def get_attention_mask(self, tokens_tensor, ignore_index):
        attention_mask = torch.where(tokens_tensor==ignore_index, torch.tensor(0., device=self.device) ,torch.tensor(1., device=self.device))
        return attention_mask

    def get_position_ids(self, tokens_tensor):
        position_ids = torch.arange(tokens_tensor.shape[1], device=self.device).expand((1, -1))
        return position_ids


class My_Classifer(nn.Module):
    def __init__(self, TAGS_fileds, input_dim, device, ignore_index=None, loss_weight=None):
        super(My_Classifer, self).__init__()
        self.to(device)
        self.device = device
        self.ignore_index = ignore_index
        self.input_dim = input_dim
        self.output_dim = len(TAGS_fileds.vocab)
        self.classfier_1 = nn.Linear(self.input_dim, int(self.input_dim/2), bias=False)
        self.classfier_2 = nn.Linear(int(self.input_dim/2), int(self.output_dim), bias=False)
        # self.CRF = CRF(self.output_dim, batch_first=True)
        self.loss_weight = loss_weight

    def forward(self, common_embedding, IF_CRF):
        # common_embedding = nn.Dropout(p=0.5)(common_embedding)
        res_1 = F.relu(self.classfier_1(common_embedding))
        res = self.classfier_2(res_1)
        # res_1 = F.relu(self.classfier_1(common_embedding)[0])
        if IF_CRF:
            res = self.CRF.decode(res)
            res = torch.Tensor(res).unsqueeze(2).to(self.device)
        return res

    def get_CRF_loss(self, res, targets):
        crf_loss = -self.CRF(res, targets, reduction='token_mean')
        return crf_loss

    def get_ce_loss(self, res, targets):
        CrossEntropy_loss = torch.nn.functional.cross_entropy(res.permute(0,2,1), targets, ignore_index=self.ignore_index, weight=self.loss_weight)
        return CrossEntropy_loss


class My_Ensembled_Classifier(nn.Module):
    def __init__(self, args, device):
        super(My_Ensembled_Classifier, self).__init__()
        self.to(device)
        self.device = device
        self.args = args

    def create_classifers(self, TAGS_Types_fileds_dic):
        self.TAGS_Types_fileds_dic = TAGS_Types_fileds_dic
        for key, TAGS_fileds in self.TAGS_Types_fileds_dic.items():
            ignore_index = TAGS_fileds[1].vocab.stoi["[PAD]"]
            my_classifer = My_Classifer(TAGS_fileds[1], self.args.Word_embedding_size, self.device, ignore_index=ignore_index)
            setattr(self, 'my_classifer_{0}'.format(key), my_classifer)

    def get_classifer(self, i):
        return getattr(self, 'my_classifer_{0}'.format(i))

    def forward(self, common_embedding, IF_CRF):
        res_list = []
        key_list = []
        res_dict = {}
        for key in self.TAGS_Types_fileds_dic.keys():
            res = self.get_classifer(key)(common_embedding, IF_CRF)
            res_dict[key] = res
            res_list.append(res)
            key_list.append(key)
        return res_dict, res_list, key_list

    def get_ensembled_CRF_loss(self, res_list, targets_list):
        crf_loss_list = []
        for i, key in enumerate(self.TAGS_Types_fileds_dic.keys()):
            crf_loss = self.get_classifer(key).get_CRF_loss(res_list[i], targets_list[i])
            crf_loss_list.append(crf_loss)
        return crf_loss_list


class My_Entity_Span_Classifier(My_Ensembled_Classifier):
    def get_ensembled_ce_loss(self, batch_pred_for_loss_sub_task_tensor, batch_gold_for_loss_sub_task_tensor, weight=None):
        pred_tensor = batch_pred_for_loss_sub_task_tensor.permute(0, 2, 1)
        target_tensor = batch_gold_for_loss_sub_task_tensor.permute(0, 1)
        ce_loss = F.cross_entropy(pred_tensor, target_tensor, weight=weight)
        return ce_loss


class My_Entity_Span_And_Type_Classifier(My_Ensembled_Classifier):
    def get_ensembled_ce_loss(self, batch_pred_for_loss_sub_task_tensor, batch_gold_for_loss_sub_task_tensor, weight=None):
        pred_tensor = batch_pred_for_loss_sub_task_tensor.permute(1, 3, 2, 0)
        target_tensor = batch_gold_for_loss_sub_task_tensor.permute(1, 2, 0)
        ce_loss = F.cross_entropy(pred_tensor, target_tensor, weight=weight)
        return ce_loss


class My_Entity_Type_Classifier(My_Ensembled_Classifier):
    def __init__(self, args, device):
        super(My_Ensembled_Classifier, self).__init__()
        self.to(device)
        self.device = device
        self.args = args

        self.TAGS_my_types_classification = torchtext.data.Field(dtype=torch.long, batch_first=True, pad_token=None, unk_token=None)
        self.TAGS_my_types_classification.vocab =  {"no":0, "yes":1}
        self.ignore_index = len(self.TAGS_my_types_classification.vocab)+1

    def create_classifers(self, TAGS_Types_fileds_dic, TAGS_sep_entity_fileds_dic):
        self.TAGS_Types_fileds_dic = TAGS_Types_fileds_dic
        self.TAGS_sep_entity_fileds_dic = TAGS_sep_entity_fileds_dic
        self.my_entity_type_sub_task_list = list(self.TAGS_Types_fileds_dic.keys())

        self.total_entity_type_to_index_dic = dict(zip(self.my_entity_type_sub_task_list, list(range(1, len(self.my_entity_type_sub_task_list)+1 ) )))
        self.total_entity_type_to_index_dic["None"] = 0

        for sub_task in self.my_entity_type_sub_task_list:
            my_classifer = My_Classifer(self.TAGS_my_types_classification, self.args.Word_embedding_size, self.device, ignore_index=self.ignore_index)
            setattr(self, 'my_classifer_{0}'.format(sub_task), my_classifer)

    def get_entity_vec_segment(self, batch_entity, common_embedding):
        """
        one_batch = output of get_entity_res_segment
        """
        batch_entity_vec_list = []
        for sent_index in range(len((batch_entity))):
            entity_vec_list = []
            for entity in batch_entity[sent_index]:
                entity_token_span_temp = []
                entity_emb_temp = []
                for token in eval(entity):
                    entity_emb_temp.append(common_embedding[sent_index][token])
                    entity_token_span_temp.append(token)

                if len(entity_emb_temp) > 0:
                    entity_average_vec = sum(entity_emb_temp) / len(entity_emb_temp)
                else:
                    entity_average_vec = None
                entity_vec_list.append(entity_average_vec)

            if entity_vec_list:
                batch_entity_vec_list.append(torch.stack(entity_vec_list))
            else:
                # valid may have no entity
                batch_entity_vec_list.append(torch.tensor([0]*(self.args.Word_embedding_size), device=self.device).float().unsqueeze(0))

        batch_entity_vec_tensor = pad_sequence(batch_entity_vec_list, batch_first=True, padding_value=self.ignore_index)

        return batch_entity_vec_tensor, batch_entity  # batch_entity_token_span_list = one_batch

    def forward(self, batch_entity_vec, IF_CRF):
        assert IF_CRF==False
        loss_list = []
        pred_entity_type_list = []
        for sub_task_index, key in enumerate(self.my_entity_type_sub_task_list):
            res = self.get_classifer(key)(batch_entity_vec, IF_CRF)
            loss_list.append(res)
            pred_entity_type_list.append(res[:,:,self.TAGS_my_types_classification.vocab["yes"]])

        # don't consider None type
        pred_entity_type = torch.argmax(torch.stack(pred_entity_type_list), dim=0)
        loss_tensor = torch.stack(loss_list)

        return pred_entity_type, loss_tensor

    def make_pred_gold_for_loss(self, batch_gold_res_list, batch_entity_token_span_list, vocab_dic):
        batch_gold_for_loss_sub_task_list = []
        for sent_index, sent_entity_pair in enumerate(batch_entity_token_span_list):
            one_sent_list = []
            for entity_pair in sent_entity_pair:
                sub_task_list = []
                for sub_task, sub_task_values_list in batch_gold_res_list[sent_index].items():
                    if eval(entity_pair) in sub_task_values_list:
                        sub_task_list.append(torch.tensor(vocab_dic["yes"], dtype=torch.long, device=self.device))
                    else:
                        sub_task_list.append(torch.tensor(vocab_dic["no"], dtype=torch.long, device=self.device))
                one_sent_list.append(torch.stack(sub_task_list))

            if not one_sent_list:
                pad_sub_task_list = [torch.tensor(self.ignore_index, dtype=torch.long, device=self.device)]*len(self.my_entity_type_sub_task_list)
                one_sent_list.append(torch.stack(pad_sub_task_list))

            batch_gold_for_loss_sub_task_list.append(torch.stack(one_sent_list))

        gold_for_loss_sub_task_tensor = pad_sequence(batch_gold_for_loss_sub_task_list, batch_first=True, padding_value=self.ignore_index)
        gold_for_loss_sub_task_tensor = gold_for_loss_sub_task_tensor.permute(2, 0, 1)

        return gold_for_loss_sub_task_tensor

    def get_ensembled_ce_loss(self, batch_pred_for_loss_sub_task_list, batch_gold_for_loss_sub_task_tensor, ignore_index, weight=None):
        pred_tensor = batch_pred_for_loss_sub_task_list.permute(1, 3, 2, 0)
        target_tensor = batch_gold_for_loss_sub_task_tensor.permute(1, 2, 0)
        ce_loss = F.cross_entropy(pred_tensor, target_tensor, ignore_index=ignore_index, weight=weight)
        return ce_loss


class My_Relation_Classifier(nn.Module):
    def __init__(self, args, tokenizer_RC, device):
        super(My_Relation_Classifier, self).__init__()
        self.to(device)
        self.device = device
        self.TAGS_my_types_classification = torchtext.data.Field(dtype=torch.long, batch_first=True, pad_token=None, unk_token=None)
        self.TAGS_my_types_classification.vocab =  {"no":0, "yes":1}
        # self.ignore_index = len(self.TAGS_my_types_classification.vocab)+1
        self.ignore_index = len(self.TAGS_my_types_classification.vocab)
        self.args = args
        self.tokenizer_RC = tokenizer_RC

    def get_classifer(self, i):
        return getattr(self, 'my_classifer_{0}'.format(i))

    def create_classifers(self, TAGS_Types_fileds_dic, TAGS_sampled_entity_span_fileds_dic, TAGS_Entity_Type_fileds_dic):
        self.TAGS_Types_fileds_dic = TAGS_Types_fileds_dic
        self.TAGS_sampled_entity_span_fileds_dic = TAGS_sampled_entity_span_fileds_dic
        self.my_relation_sub_task_list = list(self.TAGS_Types_fileds_dic.keys())
        if self.args.Entity_Prep_Way == "entity_type_embedding":
            self.entity_type_embedding = nn.Embedding(len(TAGS_Entity_Type_fileds_dic.keys())+1, self.args.Type_emb_num).to(self.device)

        if self.args.If_add_prototype or self.args.Entity_Prep_Way == "entity_type_embedding":
            self.relation_input_dim = self.args.Word_embedding_size * 2 + self.args.Type_emb_num* 2
        else:
            self.relation_input_dim = self.args.Word_embedding_size * 2

        for sub_task in self.my_relation_sub_task_list:
            my_classifer = My_Classifer(self.TAGS_my_types_classification, self.relation_input_dim, self.device, ignore_index=self.ignore_index)
            setattr(self, 'my_classifer_{0}'.format(sub_task), my_classifer)

        if self.args.Weight_Loss:
            self.yes_no_weight = torch.tensor([self.args.Min_weight, self.args.Max_weight], device=self.device)
            print("self.yes_no_weight", self.yes_no_weight)

        else:
            self.yes_no_weight = None

    def forward(self, batch_added_marker_entity_span_vec, IF_CRF):
        loss_list = []
        sub_task_res_prob_list = []
        sub_task_res_yes_no_index_list = []
        for sub_task in self.my_relation_sub_task_list:
            res = self.get_classifer(sub_task)(batch_added_marker_entity_span_vec, IF_CRF)
            loss_list.append(res)
            res = torch.max(res, 2)
            sub_task_res_prob_list.append(res[0])
            sub_task_res_yes_no_index_list.append(res[1])

        batch_pred_res_prob = torch.stack(sub_task_res_prob_list).permute(1,2,0)
        batch_pred_res_yes_no_index = torch.stack(sub_task_res_yes_no_index_list).permute(1,2,0)

        yes_flag_tensor = torch.tensor(self.TAGS_my_types_classification.vocab["yes"], device=self.device)
        no_mask_tensor = batch_pred_res_yes_no_index != yes_flag_tensor

        batch_pred_res_prob_masked = torch.masked_fill(batch_pred_res_prob, no_mask_tensor, torch.tensor(-999, device=self.device))

        # deal the solution of all results are no, there there is None classifer, commented down these two lines
        pad_tensor = torch.Tensor(batch_pred_res_prob_masked.shape[0], batch_pred_res_prob_masked.shape[1], 1).fill_(-998).to(self.device)
        batch_pred_res_prob_masked = torch.cat((batch_pred_res_prob_masked, pad_tensor), 2)

        pred_type_tensor = torch.max(batch_pred_res_prob_masked, 2)[1]

        return pred_type_tensor, loss_list

    def get_entity_pair_rep(self, entity_pair_span, sentence_embedding, Entity_type_TAGS_Types_fileds_dic, dic_map_span_type=None, entity_type_rep_dic=None):

        def prototype_information(entity_1, entity_2):
            if self.args.If_add_prototype:
                entity_1_type = dic_map_span_type[str(entity_span_1)]
                entity_2_type = dic_map_span_type[str(entity_span_2)]

                entity_1 = torch.cat((entity_1, entity_type_rep_dic[entity_1_type].data))
                entity_2 = torch.cat((entity_2, entity_type_rep_dic[entity_2_type].data))
            return entity_1, entity_2


        entity_span_1, entity_span_2 = entity_pair_span
        entity_1_head = entity_span_1[0]
        entity_1_tail = entity_span_1[-1]
        entity_2_head = entity_span_2[0]
        entity_2_tail = entity_span_2[-1]


        if self.args.Entity_Prep_Way== "entitiy_type_marker":
            entity_1 = sentence_embedding[entity_1_head]
            entity_2 = sentence_embedding[entity_2_head]
            entity_1, entity_2 = prototype_information(entity_1, entity_2)
        elif self.args.Entity_Prep_Way == "standard":
            entity_1 = torch.sum(sentence_embedding[entity_1_head:entity_1_tail], dim=0).squeeze()
            entity_2 = torch.sum(sentence_embedding[entity_2_head:entity_2_tail], dim=0).squeeze()
            entity_1, entity_2 = prototype_information(entity_1, entity_2)
        elif self.args.Entity_Prep_Way ==  "entity_type_embedding":
            entity_1 = torch.sum(sentence_embedding[entity_1_head:entity_1_tail], dim=0).squeeze()
            entity_2 = torch.sum(sentence_embedding[entity_2_head:entity_2_tail], dim=0).squeeze()

            type_1 = dic_map_span_type[str(entity_span_1)]
            type_2 = dic_map_span_type[str(entity_span_2)]
            type_1_index = torch.tensor([[Entity_type_TAGS_Types_fileds_dic[type_1]]], device=self.device).long()
            type_2_index = torch.tensor([[Entity_type_TAGS_Types_fileds_dic[type_2]]], device=self.device).long()


            entity_1_type_embedding = self.entity_type_embedding(type_1_index).squeeze()
            entity_2_type_embedding = self.entity_type_embedding(type_2_index).squeeze()
            entity_1 = torch.cat((entity_1, entity_1_type_embedding))
            entity_2 = torch.cat((entity_2, entity_2_type_embedding))

            entity_1, entity_2 = prototype_information(entity_1, entity_2)

        else:
            raise Exception("args.Entity_Prep_Way error !")

        entity_pair_rep = torch.cat((entity_1, entity_2))

        # if self.args.Pair_Combine_Way== "cat":
        #     entity_pair_rep = torch.cat(entity_1, entity_2)
        # elif self.args.Pair_Combine_Way == "mentions_pooling":
        #     max_index = max([entity_1_head, entity_1_tail, entity_2_head, entity_2_tail])
        #     min_index = min([entity_1_head, entity_1_tail, entity_2_head, entity_2_tail])
        #     pooling = nn.AdaptiveMaxPool1d(2)
        #     entity_pair_rep = pooling(sentence_embedding[min_index:max_index].permute(1,0).unsqueeze(0)).squeeze()
        #     entity_pair_rep = torch.cat((entity_pair_rep[:,0], entity_pair_rep[:,1]))
        # else:
        #     raise Exception("args.Pair_Combine_Way error !")

        return entity_pair_rep

    def get_entity_pair_span_vec(self, batch_entity, batch_tokens, batch_entity_type, entity_type_rep_dic, bert_RC, Entity_type_TAGS_Types_fileds_dic, NER_last_embedding):
        padding_value = self.tokenizer_RC.vocab['[PAD]']
        if self.args.If_soft_share:
            encoder_hidden_states= NER_last_embedding
        else:
            encoder_hidden_states=None

        if self.args.Entity_Prep_Way == "entitiy_type_marker":
            raw_batch_entity = copy.deepcopy(batch_entity)
            batch_tokens_marker = []
            for sent_index, (one_sent_tokens, one_sent_span, one_sent_type)  in enumerate(zip(batch_tokens, batch_entity, batch_entity_type)):
                temp_token_list = one_sent_tokens.tolist()
                for entity_index, (span, type) in enumerate(zip(one_sent_span, one_sent_type)):
                    temp_token_list.insert(span[0], self.tokenizer_RC.convert_tokens_to_ids("[Entity_"+type+"]"))
                    temp_token_list.insert(span[-1]+2, self.tokenizer_RC.convert_tokens_to_ids("[/Entity_"+type+"]"))

                    batch_entity[sent_index][entity_index].insert(0,span[0])
                    batch_entity[sent_index][entity_index].append(span[-1]+2)

                    # need batch_span is in order in entity list
                    batch_entity[sent_index][entity_index+1:] = [ [j+2 for j in i] for i in batch_entity[sent_index][entity_index+1:]]
                    batch_entity[sent_index][entity_index][1:-1] = [i+1 for i in batch_entity[sent_index][entity_index][1:-1]]
                batch_tokens_marker.append(torch.tensor(temp_token_list, device=self.device))

            common_embedding = bert_RC(pad_sequence(batch_tokens_marker, batch_first=True, padding_value=padding_value), encoder_hidden_states=encoder_hidden_states)[0]

            batch_entity_pair_span_list = []
            batch_entity_pair_vec_list = []
            batch_sent_len_list = []

            for sent_index, (raw_one_sent_entity, one_sent_entity, one_sent_type)  in enumerate(zip(raw_batch_entity, batch_entity, batch_entity_type)):
                if self.args.If_add_prototype:
                    dic_map_span_type = {}
                    for span, type in zip(one_sent_entity, one_sent_type):
                        dic_map_span_type[str(span)] = type

                sent_entity_pair_span_list = list(combinations(one_sent_entity, 2))
                sent_entity_pair_span_list = [sorted(i) for i in sent_entity_pair_span_list]
                batch_sent_len_list.append(len(sent_entity_pair_span_list))

                raw_sent_entity_pair_span_list = list(combinations(raw_one_sent_entity, 2))
                raw_sent_entity_pair_span_list = [sorted(i) for i in raw_sent_entity_pair_span_list]
                batch_entity_pair_span_list.append(raw_sent_entity_pair_span_list)

                sent_entity_pair_rep_list = []
                if sent_entity_pair_span_list:
                    for entity_pair_span in sent_entity_pair_span_list:
                        if self.args.If_add_prototype:
                            one_sent_rep = self.get_entity_pair_rep(entity_pair_span, common_embedding[sent_index], Entity_type_TAGS_Types_fileds_dic,
                                                                    dic_map_span_type, entity_type_rep_dic)
                        else:
                            one_sent_rep = self.get_entity_pair_rep(entity_pair_span, common_embedding[sent_index], Entity_type_TAGS_Types_fileds_dic)
                        sent_entity_pair_rep_list.append(one_sent_rep)
                else:
                    sent_entity_pair_rep_list.append(torch.tensor([0]* self.relation_input_dim).float().to(self.device))

                batch_entity_pair_vec_list.append(torch.stack(sent_entity_pair_rep_list))

            batch_added_marker_entity_span_vec = pad_sequence(batch_entity_pair_vec_list, batch_first=True, padding_value=padding_value)

            return batch_added_marker_entity_span_vec, batch_entity_pair_span_list, batch_sent_len_list

        elif self.args.Entity_Prep_Way == "standard" or "entity_type_embedding":
            common_embedding = bert_RC(batch_tokens, encoder_hidden_states=encoder_hidden_states, ignore_index = padding_value)[0]
            batch_entity_pair_span_list = []
            batch_entity_pair_vec_list = []
            batch_sent_len_list = []

            for sent_index, (one_sent_entity, one_sent_type)  in enumerate(zip(batch_entity, batch_entity_type)):
                if self.args.If_add_prototype or self.args.Entity_Prep_Way == "entity_type_embedding":
                    dic_map_span_type = {}
                    for span, type in zip(one_sent_entity, one_sent_type):
                        dic_map_span_type[str(span)] = type

                sent_entity_pair_span_list = list(combinations(one_sent_entity, 2))
                batch_sent_len_list.append(len(sent_entity_pair_span_list))
                sent_entity_pair_span_list = [sorted(i) for i in sent_entity_pair_span_list]

                sent_entity_pair_rep_list = []
                if sent_entity_pair_span_list:
                    for entity_pair_span in sent_entity_pair_span_list:
                        if self.args.If_add_prototype or self.args.Entity_Prep_Way == "entity_type_embedding":
                            one_sent_rep = self.get_entity_pair_rep(entity_pair_span, common_embedding[sent_index], Entity_type_TAGS_Types_fileds_dic,
                                           dic_map_span_type, entity_type_rep_dic)
                        else:
                            one_sent_rep = self.get_entity_pair_rep(entity_pair_span, common_embedding[sent_index], Entity_type_TAGS_Types_fileds_dic)
                        sent_entity_pair_rep_list.append(one_sent_rep)
                else:
                    sent_entity_pair_rep_list.append(torch.tensor([0]* self.relation_input_dim).float().to(self.device))

                batch_entity_pair_vec_list.append(torch.stack(sent_entity_pair_rep_list))
                batch_entity_pair_span_list.append(sent_entity_pair_span_list)
            batch_added_marker_entity_span_vec = pad_sequence(batch_entity_pair_vec_list, batch_first=True, padding_value=padding_value)

            return batch_added_marker_entity_span_vec, batch_entity_pair_span_list, batch_sent_len_list
        else:
            raise Exception("Entity_Prep_Way wrong !")

    def make_gold_for_loss(self, gold_all_sub_task_res_list, batch_entity_span_list, vocab_dic):
        batch_gold_for_loss_sub_task_list = []
        for sent_index, sent_entity_pair in enumerate(batch_entity_span_list):
            one_sent_list = []
            for entity_pair in sent_entity_pair:
                sub_task_list = []
                for sub_task, sub_task_values_list in gold_all_sub_task_res_list[sent_index].items():
                    if entity_pair in sub_task_values_list:
                        sub_task_list.append(torch.tensor(vocab_dic["yes"], dtype=torch.long, device=self.device))
                    else:
                        sub_task_list.append(torch.tensor(vocab_dic["no"], dtype=torch.long, device=self.device))
                one_sent_list.append(torch.stack(sub_task_list))

            if not one_sent_list:
                pad_sub_task_list = [torch.tensor(self.ignore_index, dtype=torch.long, device=self.device)]*len(self.my_relation_sub_task_list)
                one_sent_list.append(torch.stack(pad_sub_task_list))

            batch_gold_for_loss_sub_task_list.append(torch.stack(one_sent_list))

        gold_for_loss_sub_task_tensor = pad_sequence(batch_gold_for_loss_sub_task_list, batch_first=True, padding_value=self.ignore_index)
        gold_for_loss_sub_task_tensor = gold_for_loss_sub_task_tensor.permute(2, 0, 1)

        return gold_for_loss_sub_task_tensor

    # def get_ensembled_ce_loss(self, batch_pred_for_loss_sub_task_list, batch_gold_for_loss_sub_task_tensor):
    #     pred_tensor = torch.stack(batch_pred_for_loss_sub_task_list).permute(1, 3, 2, 0)
    #     target_tensor = batch_gold_for_loss_sub_task_tensor.permute(1, 2, 0)
    #     ce_loss = F.cross_entropy(pred_tensor, target_tensor, ignore_index=self.ignore_index, weight=self.loss_weight)
    #     return ce_loss
    #
    # def BCE_loss(self, batch_pred_for_loss_sub_task_list, batch_gold_for_loss_sub_task_tensor):
    #     pred_tensor = torch.stack(batch_pred_for_loss_sub_task_list)
    #     target_tensor = torch.where(batch_gold_for_loss_sub_task_tensor==3, torch.tensor(0, device=self.device), batch_gold_for_loss_sub_task_tensor)
    #     target_tensor = torch.zeros(pred_tensor.shape, device=self.device).scatter_(3, target_tensor.unsqueeze(3), 1)
    #
    #     criterion = nn.BCEWithLogitsLoss(pos_weight=self.loss_weight)
    #     bec_loss = criterion(pred_tensor, target_tensor)
    #     return bec_loss
    #
    # def crossentropy_few_shot(self, batch_pred_for_loss_sub_task_list, batch_gold_for_loss_sub_task_tensor):
    #     """
    #     y_pred don't need softmax
    #     """
    #     pred_tensor = torch.stack(batch_pred_for_loss_sub_task_list).permute(1, 2, 0, 3)
    #     target_tensor = batch_gold_for_loss_sub_task_tensor.permute(1, 0, 2)
    #     prior = torch.tensor(self.yes_no_relation_list, device=self.device)
    #
    #     log_prior = torch.log(prior + 1e-8)
    #     y_pred = pred_tensor + 1.0 * log_prior
    #     fel_ce_loss = F.cross_entropy(y_pred.permute(0,3,2,1), target_tensor, ignore_index=self.ignore_index, weight=self.loss_weight)
    #     return fel_ce_loss

    def get_ensembled_ce_loss(self, batch_pred_for_loss_sub_task_list, batch_gold_for_loss_sub_task_tensor):
        loss_list = []
        for sub_task_index, sub_task_batch_pred in enumerate(batch_pred_for_loss_sub_task_list):
            sub_task_batch_gold = batch_gold_for_loss_sub_task_tensor[sub_task_index]
            ce_loss = F.cross_entropy(sub_task_batch_pred.permute(0,2,1), sub_task_batch_gold, ignore_index=self.ignore_index, weight=self.yes_no_weight, reduction='mean')
            # loss_list.append(ce_loss)
            loss_list.append(ce_loss)
        return sum(loss_list)/len(loss_list)

    def BCE_loss(self, batch_pred_for_loss_sub_task_list, batch_gold_for_loss_sub_task_tensor):
        loss_list = []
        for sub_task_index, sub_task_batch_pred in enumerate(batch_pred_for_loss_sub_task_list):
            sub_task_batch_gold = batch_gold_for_loss_sub_task_tensor[sub_task_index]
            target_tensor = torch.where(sub_task_batch_gold==self.ignore_index, torch.tensor(0, device=self.device), sub_task_batch_gold)
            target_tensor = torch.zeros(sub_task_batch_pred.shape, device=self.device).scatter_(2, target_tensor.unsqueeze(2), 1)

            criterion = nn.BCEWithLogitsLoss(pos_weight=self.yes_no_weight, reduction='mean')
            bec_loss = criterion(sub_task_batch_pred, target_tensor)
            # loss_list.append(bec_loss)
            loss_list.append(bec_loss)
        return sum(loss_list)/len(loss_list)

    def crossentropy_few_shot(self, batch_pred_for_loss_sub_task_list, batch_gold_for_loss_sub_task_tensor):
        loss_list = []
        for sub_task_index, sub_task_batch_pred in enumerate(batch_pred_for_loss_sub_task_list):
            target_tensor = batch_gold_for_loss_sub_task_tensor[sub_task_index]
            prior = torch.tensor(self.FSL_weight[sub_task_index], device=self.device)
            log_prior = torch.log(prior + 1e-8)
            y_pred = sub_task_batch_pred + 1.0 * log_prior
            fel_ce_loss = F.cross_entropy(y_pred.permute(0,2,1), target_tensor, ignore_index=self.ignore_index, weight=self.loss_weight[sub_task_index], reduction='mean')
            loss_list.append(fel_ce_loss)
        return sum(loss_list)/len(loss_list)


class My_Model(nn.Module):
    def __init__(self, bert_list, args, device):
        super(My_Model, self).__init__()
        self.to(device)
        self.device = device
        self.bert_NER = bert_list[0]
        if "relation" in args.Task_list:
            self.bert_RC = bert_list[1]
        self.task_list = args.Task_list
        self.IF_CRF =  args.IF_CRF
        self.args = args
        if self.args.Share_embedding:
            # self.bert_RC.bert.embeddings.word_embeddings.weight.data = copy.deepcopy(self.bert_NER.bert.embeddings.word_embeddings.weight.data).detach()
            self.bert_RC.bert.embeddings.word_embeddings = self.bert_NER.bert.embeddings.word_embeddings

    def add_classifers(self, classifiers_dic, task_list):
        self.classifiers_dic = {}
        for task in task_list:
            setattr(self, 'my_{0}_classifier'.format(task), classifiers_dic[task])
            self.classifiers_dic[task] = classifiers_dic[task]

    def get_entity_type_data(self, batch, common_embedding, valid_test_flag, dic_res_one_batch, epoch):
        if valid_test_flag == 'train':
            if epoch<5:
                gold_input_ratio = (50-epoch)/100
            elif epoch<10:
                gold_input_ratio = (30-epoch)/100
            elif epoch<15:
                gold_input_ratio = (10-epoch)/100
            else:
                gold_input_ratio = -1

            random_ratio = np.random.uniform(0,1)

            if gold_input_ratio > random_ratio:
                TAGS_filed = self.my_entity_type_classifier.TAGS_sep_entity_fileds_dic["sep_entity"][1]
                batch_entity = get_entity_res_segment(batch.sep_entity, TAGS_filed, need_sep_flag=False)
            else:
                TAGS_filed = self.my_entity_span_classifier.TAGS_Types_fileds_dic["entity_span"][1]
                batch_pred_entity_res = dic_res_one_batch["entity_span"][1]
                batch_entity = []
                for sent_entity_list in get_entity_res_segment(batch_pred_entity_res, TAGS_filed, need_sep_flag=True):
                    temp_list = []
                    for entity in sent_entity_list:
                        temp_list.append(str([i[0] for i in entity]))
                    batch_entity.append(temp_list)
        else:
            TAGS_filed = self.my_entity_span_classifier.TAGS_Types_fileds_dic["entity_span"][1]
            batch_pred_entity_res = dic_res_one_batch["entity_span"][1]
            batch_entity = []
            for sent_entity_list in get_entity_res_segment(batch_pred_entity_res, TAGS_filed, need_sep_flag=True):
                temp_list = []
                for entity in sent_entity_list:
                    temp_list.append(str([i[0] for i in entity]))
                batch_entity.append(temp_list)

        batch_entity_vec_list, batch_entity_token_span_list = self.my_entity_type_classifier.get_entity_vec_segment(batch_entity, common_embedding)

        return batch_entity_vec_list, batch_entity_token_span_list

    def get_relation_data(self, batch, valid_test_flag, dic_res_one_batch, epoch):
        batch_entity = []
        batch_entity_type = []

        def inner_realtion_data_gold_span():
            batch_pred_entity_res = batch.sampled_entity_span
            TAGS_filed = self.my_relation_classifier.TAGS_sampled_entity_span_fileds_dic["sampled_entity_span"][1]

            for sent_index, one_sent_entity in enumerate(batch_pred_entity_res):
                one_sent_temp_list = one_sent_entity[:get_sent_len(one_sent_entity, TAGS_filed)]    # deal with PAD
                one_sent_temp_list = [eval(TAGS_filed.vocab.itos[int(i)]) for i in one_sent_temp_list.cpu().numpy().tolist()]
                batch_entity.append(sorted(one_sent_temp_list, key=lambda s: s[0]))

                type_dic = dic_res_one_batch['entity_type'][0][sent_index]
                batch_entity_type.append(self.provide_dic_entity_type(one_sent_temp_list, type_dic))

        def inner_realtion_data_pred_span():
            batch_pred_entity_res = dic_res_one_batch["entity_span"][1]
            TAGS_filed = self.my_entity_span_classifier.TAGS_Types_fileds_dic["entity_span"][1]

            for sent_index, sent_entity_list in enumerate(get_entity_res_segment(batch_pred_entity_res, TAGS_filed, need_sep_flag=True)):
                one_sent_all_type_span_list = []
                for entity in sent_entity_list:
                    one_sent_all_type_span_list.append([i[0] for i in entity])

                one_sent_temp_list = sorted(one_sent_all_type_span_list, key=lambda s: s[0])
                batch_entity.append(one_sent_temp_list)

                type_dic = dic_res_one_batch['entity_type'][1][sent_index]
                batch_entity_type.append(self.provide_dic_entity_type(one_sent_temp_list, type_dic))

        def inner_realtion_data_pred_span_and_type():
            raise Exception("There is a bug no fixed, don't use it")
            batch_pred_entity_res_dic = dic_res_one_batch["entity_span_and_type"][1]

            for sent_index in range(len(batch)):
                one_sent_all_type_span_list = []
                for index, (key, sent_entity_list) in enumerate(batch_pred_entity_res_dic.items()):
                    for entity_pair in sent_entity_list[sent_index]:
                        one_sent_all_type_span_list.append([i[0] for i in entity_pair])

                one_sent_temp_list = sorted(one_sent_all_type_span_list, key=lambda s: s[0])
                batch_entity.append(one_sent_temp_list)

                type_dic = dic_res_one_batch['entity_type'][1][sent_index]
                batch_entity_type.append(self.provide_dic_entity_type(one_sent_temp_list, type_dic))

        if valid_test_flag == 'train':
            if epoch<5:
                gold_input_ratio = (50-epoch)/100
            elif epoch<10:
                gold_input_ratio = (30-epoch)/100
            elif epoch<15:
                gold_input_ratio = (10-epoch)/100
            else:
                gold_input_ratio = -1

            random_ratio = np.random.uniform(0,1)
            # gold_input_ratio = 10

            if gold_input_ratio > random_ratio:
                if self.args.Relation_input=="entity_span" or self.args.Relation_input=="entity_span_and_type":
                    inner_realtion_data_gold_span()
                else:
                    raise Exception("relation_input args wrong !")
            else:
                if self.args.Relation_input=="entity_span":
                    inner_realtion_data_pred_span()
                elif self.args.Relation_input=="entity_span_and_type":
                    inner_realtion_data_pred_span_and_type()
                else:
                    raise Exception("relation_input args wrong !")
        else:
            if self.args.Only_relation:
                inner_realtion_data_gold_span()
            else:
                if self.args.Relation_input == "entity_span":
                    inner_realtion_data_pred_span()
                elif self.args.Relation_input == "entity_span_and_type":
                    inner_realtion_data_pred_span_and_type()
                else:
                    raise Exception("relation_input args wrong !")

        assert len(batch_entity)==len(batch_entity_type)
        return batch_entity, batch_entity_type

    def provide_dic_entity_type(self, entity_list, type_dic):
        entity_type_list = []
        for entity in entity_list :
            add_flag = False
            for k,v in type_dic.items():
                if entity in v:
                    entity_type_list.append(k)
                    add_flag = True
                    break
            if not add_flag:
                entity_type_list.append("None")
        assert len(entity_type_list)==len(entity_list)
        return entity_type_list

    def forward(self, batch_list, epoch, entity_type_rep_dic, valid_test_flag='train'):
        dic_res_one_batch ={}
        dic_loss_one_batch ={}

        batch_NER = batch_list[0]
        RC_common_embedding, common_embedding_NER = self.bert_NER(batch_NER.tokens)

        if  "entity_span" in self.task_list:
            batch_gold_and_pred_entity_res, entity_span_batch_loss = self.entity_span_extraction(batch_NER, common_embedding_NER)
            dic_res_one_batch["entity_span"] = batch_gold_and_pred_entity_res
            dic_loss_one_batch["entity_span"] = entity_span_batch_loss

        if "entity_type" in self.task_list:
            batch_entity_vec_list, batch_entity_token_span_list = self.get_entity_type_data(batch_NER, common_embedding_NER, valid_test_flag, dic_res_one_batch, epoch)
            batch_gold_and_pred_entity_type_res, one_batch_entity_type_loss = self.entity_type_extraction(batch_NER, batch_entity_vec_list, batch_entity_token_span_list)
            dic_res_one_batch["entity_type"] = batch_gold_and_pred_entity_type_res
            dic_loss_one_batch["entity_type"] = one_batch_entity_type_loss

        if "entity_span_and_type" in self.task_list:
            sub_task_batch_gold_and_pred_entity_span_and_type_res, batch_entity_span_and_type_loss = self.entity_span_and_type_extraction(batch_NER, common_embedding_NER)
            dic_res_one_batch["entity_span_and_type"] = sub_task_batch_gold_and_pred_entity_span_and_type_res
            dic_loss_one_batch["entity_span_and_type"] = batch_entity_span_and_type_loss

        if "relation" in self.task_list:
            batch_RC = batch_list[1]

            batch_entity, batch_entity_type = self.get_relation_data(batch_RC, valid_test_flag, dic_res_one_batch, epoch)
            one_batch_relation_res, one_batch_relation_loss = self.relation_extraction(batch_RC, batch_entity, batch_entity_type, entity_type_rep_dic, common_embedding_NER)
            dic_res_one_batch["relation"] = one_batch_relation_res
            dic_loss_one_batch["relation"] = one_batch_relation_loss

        return dic_loss_one_batch, dic_res_one_batch

    def entity_span_extraction(self, batch, common_embedding):
        """ Entity span extraction (for only span, my_ensembled_relation_classifier = one classifier) """
        res_dict, res_list, sub_task = self.my_entity_span_classifier(common_embedding, self.IF_CRF)
        if self.IF_CRF:
            entity_span_batch_loss = self.my_entity_span_classifier.get_ensembled_CRF_loss(res_list, [batch.entity_span])[0]
            one_batch_pred_res = res_list[0].squeeze(2)
        else:
            entity_span_batch_loss = self.my_entity_span_classifier.get_ensembled_ce_loss(res_list[0], batch.entity_span)
            one_batch_pred_res = torch.argmax(res_list[0], 2)
        return (batch.entity_span.tolist(), one_batch_pred_res), entity_span_batch_loss

    def entity_type_extraction(self, batch, batch_entity_vec_list, batch_entity_token_span_list):

        batch_pred_raw_res_list, batch_pred_for_loss_sub_task_tensor = self.my_entity_type_classifier(batch_entity_vec_list, self.IF_CRF)

        batch_pred_res_list = []
        for sent_index in range(len(batch)):
            pred_one_sent_all_sub_task_res_dic = {}
            for sub_task_index, sub_task in enumerate(self.my_entity_type_classifier.my_entity_type_sub_task_list):
                pred_one_sent_all_sub_task_res_dic.setdefault(sub_task, [])
                for entity_span, pred_type, entity_vec in zip(batch_entity_token_span_list[sent_index], batch_pred_raw_res_list[sent_index], batch_entity_vec_list[sent_index]):
                    if pred_type == sub_task_index:
                        pred_one_sent_all_sub_task_res_dic[sub_task].append(eval(entity_span))
            batch_pred_res_list.append(pred_one_sent_all_sub_task_res_dic)

        batch_gold_res_list = []
        for sent_index in range(len(batch)):
            gold_one_sent_all_sub_task_res_dic = {}
            for sub_task in self.my_entity_type_classifier.my_entity_type_sub_task_list:
                gold_one_sent_all_sub_task_res_dic.setdefault(sub_task, [])
                for entity_pair in getattr(batch, sub_task)[sent_index]:
                    entity_pair_span = self.my_entity_type_classifier.TAGS_Types_fileds_dic[sub_task][1].vocab.itos[entity_pair]
                    if str(entity_pair_span) != '[PAD]':
                        temp_pair = sorted(eval(entity_pair_span))
                        if (temp_pair not in gold_one_sent_all_sub_task_res_dic[sub_task]):
                            gold_one_sent_all_sub_task_res_dic[sub_task].append(temp_pair)
            batch_gold_res_list.append(gold_one_sent_all_sub_task_res_dic)

        batch_gold_for_loss_sub_task_tensor = self.my_entity_type_classifier.make_pred_gold_for_loss(
            batch_gold_res_list, batch_entity_token_span_list,
            self.my_entity_type_classifier.TAGS_my_types_classification.vocab)

        one_batch_entity_type_loss = self.my_entity_type_classifier.get_ensembled_ce_loss(
            batch_pred_for_loss_sub_task_tensor, batch_gold_for_loss_sub_task_tensor, self.my_entity_type_classifier.ignore_index, weight=None)

        batch_gold_and_pred_entity_type_res = (batch_gold_res_list, batch_pred_res_list)
        return batch_gold_and_pred_entity_type_res, one_batch_entity_type_loss

    def entity_span_and_type_extraction(self, batch, common_embedding):

        res_dict, res_list, sub_task_list = self.my_entity_span_and_type_classifier(common_embedding, self.IF_CRF)
        if self.IF_CRF:
            one_batch_res = [i.squeeze(0).squeeze(1) for i in res_list]
        else:
            one_batch_res = [torch.argmax(i, 2) for i in res_list]

        batch_pred_entity_joint_type_res = {}
        batch_gold_entity_joint_type_loss = []
        batch_gold_entity_joint_type_res = {}

        for sub_task_index, sub_task in enumerate(sub_task_list):
            batch_pred = get_entity_res_segment(one_batch_res[sub_task_index], self.my_entity_span_and_type_classifier.TAGS_Types_fileds_dic[sub_task][1], need_sep_flag=True)
            batch_pred_entity_joint_type_res[sub_task] = batch_pred
            batch_gold_entity_joint_type_loss.append(getattr(batch, sub_task))
            batch_gold_entity_joint_type_res[sub_task] = getattr(batch, sub_task)

        one_batch_entity_span_and_type_loss = self.my_entity_span_and_type_classifier.get_ensembled_ce_loss(torch.stack(res_list), torch.stack(batch_gold_entity_joint_type_loss))
        sub_task_batch_gold_and_pred_entity_joint_type_res = (batch_gold_entity_joint_type_res, batch_pred_entity_joint_type_res)

        return sub_task_batch_gold_and_pred_entity_joint_type_res, one_batch_entity_span_and_type_loss

    def relation_extraction(self, batch, batch_entity, batch_entity_type, entity_type_rep_dic, NER_last_embedding):
        """ Relation extraction """

        batch_added_marker_entity_vec, batch_entity_pair_span_list, batch_sent_len_list = \
            self.my_relation_classifier.get_entity_pair_span_vec(batch_entity, batch.tokens, batch_entity_type, entity_type_rep_dic,
                                                                 self.bert_RC, self.my_entity_type_classifier.total_entity_type_to_index_dic,
                                                                 NER_last_embedding)

        batch_pred_raw_res_list, batch_pred_for_loss_sub_task_list = self.my_relation_classifier(batch_added_marker_entity_vec, self.IF_CRF)

        batch_gold_res_list = []
        for sent_index in range(len(batch)):
            gold_one_sent_all_sub_task_res_dic = {}
            for sub_task in self.my_relation_classifier.my_relation_sub_task_list:
                gold_one_sent_all_sub_task_res_dic.setdefault(sub_task, [])

                for entity_pair in getattr(batch, sub_task)[sent_index]:
                    entity_pair_span = self.my_relation_classifier.TAGS_Types_fileds_dic[sub_task][1].vocab.itos[entity_pair]
                    if entity_pair_span != "[PAD]":
                        temp_pair = sorted(eval(entity_pair_span))
                        if (temp_pair not in gold_one_sent_all_sub_task_res_dic[sub_task]):
                            gold_one_sent_all_sub_task_res_dic[sub_task].append(temp_pair)

            batch_gold_res_list.append(gold_one_sent_all_sub_task_res_dic)

        batch_pred_res_list = []
        for sent_index in range(len(batch)):
            sent_len = batch_sent_len_list[sent_index]
            pred_one_sent_all_sub_task_res_dic = {}
            for sub_task_index, sub_task in enumerate(self.my_relation_classifier.my_relation_sub_task_list):
                pred_one_sent_all_sub_task_res_dic.setdefault(sub_task, [])
                for entity_pair_span, pred_type in zip(batch_entity_pair_span_list[sent_index][:sent_len], batch_pred_raw_res_list[sent_index][:sent_len]):
                    if pred_type == sub_task_index:
                        pred_one_sent_all_sub_task_res_dic[sub_task].append(entity_pair_span)
            batch_pred_res_list.append(pred_one_sent_all_sub_task_res_dic)

        # if valid_test_flag=="train":
        batch_gold_for_loss_sub_task_tensor = self.my_relation_classifier.make_gold_for_loss(
            batch_gold_res_list, batch_entity_pair_span_list, self.my_relation_classifier.TAGS_my_types_classification.vocab)

        if self.my_relation_classifier.args.Loss == "FSL":
            one_batch_relation_loss = self.my_relation_classifier.crossentropy_few_shot(batch_pred_for_loss_sub_task_list, batch_gold_for_loss_sub_task_tensor)
        elif self.my_relation_classifier.args.Loss == "CE":
            one_batch_relation_loss = self.my_relation_classifier.get_ensembled_ce_loss(batch_pred_for_loss_sub_task_list, batch_gold_for_loss_sub_task_tensor)
        elif self.my_relation_classifier.args.Loss == "BCE":
            one_batch_relation_loss = self.my_relation_classifier.BCE_loss(batch_pred_for_loss_sub_task_list, batch_gold_for_loss_sub_task_tensor)
        else:
            raise Exception("Choose loss error !")
        # else:
        #     one_batch_relation_loss = torch.tensor(0)

        one_batch_relation_res = (batch_gold_res_list, batch_pred_res_list)

        return one_batch_relation_res, one_batch_relation_loss

