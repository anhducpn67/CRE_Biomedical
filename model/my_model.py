import torch.nn as nn
from utils import get_sent_len


class MyModel(nn.Module):
    def __init__(self, encoder, classifier, args, device):
        super(MyModel, self).__init__()
        self.to(device)
        self.args = args
        self.encoder = encoder
        self.classifier = classifier

    def get_relation_data(self, batch):
        batch_entity = []
        batch_entity_type = []

        batch_entity_type_gold = self.entity_type_extraction(batch)
        TAGS_field = self.classifier.TAGS_sep_entity_fields_dic["sep_entity"][1]

        for sent_index, one_sent_entity in enumerate(batch.sep_entity):
            one_sent_temp_list = one_sent_entity[:get_sent_len(one_sent_entity, TAGS_field)]  # deal with PAD
            one_sent_temp_list = [eval(TAGS_field.vocab.itos[int(i)]) for i in one_sent_temp_list.cpu().numpy().tolist()]
            batch_entity.append(sorted(one_sent_temp_list, key=lambda s: s[0]))

            type_dic = batch_entity_type_gold[sent_index]

            batch_entity_type.append(self.provide_dic_entity_type(one_sent_temp_list, type_dic))

        assert len(batch_entity) == len(batch_entity_type)
        return batch_entity, batch_entity_type

    def provide_dic_entity_type(self, entity_list, type_dic):
        entity_type_list = []
        for entity in entity_list:
            add_flag = False
            for k, v in type_dic.items():
                if entity in v:
                    entity_type_list.append(k)
                    add_flag = True
                    break
            if not add_flag:
                entity_type_list.append("None")
        assert len(entity_type_list) == len(entity_list)
        return entity_type_list

    def forward(self, batch):
        batch_entity, batch_entity_type = self.get_relation_data(batch)
        batch_res, batch_loss = self.relation_extraction(batch, batch_entity, batch_entity_type)
        return batch_loss, batch_res

    def entity_type_extraction(self, batch):
        batch_entity_type_gold = []
        for sent_index in range(len(batch)):
            gold_one_sent_all_sub_task_res_dic = {}
            for entity_type in self.classifier.entity_type_list:
                gold_one_sent_all_sub_task_res_dic.setdefault(entity_type, [])
                for entity in getattr(batch, entity_type)[sent_index]:
                    entity_span = self.classifier.TAGS_Entity_Type_fields_dic[entity_type][1].vocab.itos[entity]
                    if str(entity_span) != '[PAD]':
                        temp_pair = sorted(eval(entity_span))
                        if temp_pair not in gold_one_sent_all_sub_task_res_dic[entity_type]:
                            gold_one_sent_all_sub_task_res_dic[entity_type].append(temp_pair)
            batch_entity_type_gold.append(gold_one_sent_all_sub_task_res_dic)

        return batch_entity_type_gold

    def relation_extraction(self, batch, batch_entity, batch_entity_type):
        """ Relation extraction """

        batch_added_marker_entity_vec, batch_entity_pair_span_list, batch_sent_len_list = \
            self.encoder.batch_get_entity_pair_rep(batch.tokens, batch_entity, batch_entity_type)

        batch_pred_raw_res_list, batch_pred_for_loss_sub_task_list = self.classifier(batch_added_marker_entity_vec)

        batch_gold_res_list = []
        for sent_index in range(len(batch)):
            gold_one_sent_all_sub_task_res_dic = {}
            for relation in self.classifier.relation_list:
                gold_one_sent_all_sub_task_res_dic.setdefault(relation, [])

                for entity_pair in getattr(batch, relation)[sent_index]:
                    entity_pair_span = self.classifier.TAGS_Types_fields_dic[relation][1].vocab.itos[
                        entity_pair]
                    if entity_pair_span != "[PAD]":
                        temp_pair = sorted(eval(entity_pair_span))
                        if temp_pair not in gold_one_sent_all_sub_task_res_dic[relation]:
                            gold_one_sent_all_sub_task_res_dic[relation].append(temp_pair)

            batch_gold_res_list.append(gold_one_sent_all_sub_task_res_dic)

        batch_pred_res_list = []
        for sent_index in range(len(batch)):
            sent_len = batch_sent_len_list[sent_index]
            pred_one_sent_all_sub_task_res_dic = {}
            for relation_index, relation in enumerate(self.classifier.relation_list):
                pred_one_sent_all_sub_task_res_dic.setdefault(relation, [])
                for entity_pair_span, pred_type in zip(batch_entity_pair_span_list[sent_index][:sent_len],
                                                       batch_pred_raw_res_list[sent_index][:sent_len]):
                    if pred_type == relation_index:
                        pred_one_sent_all_sub_task_res_dic[relation].append(entity_pair_span)
            batch_pred_res_list.append(pred_one_sent_all_sub_task_res_dic)

        batch_gold_for_loss_sub_task_tensor = self.classifier.make_gold_for_loss(
            batch_gold_res_list, batch_entity_pair_span_list,
            self.classifier.TAGS_my_types_classification.vocab)

        if self.classifier.args.Loss == "CE":
            one_batch_relation_loss = self.classifier.get_ensembled_ce_loss(batch_pred_for_loss_sub_task_list,
                                                                            batch_gold_for_loss_sub_task_tensor)
        elif self.classifier.args.Loss == "BCE":
            one_batch_relation_loss = self.classifier.BCE_loss(batch_pred_for_loss_sub_task_list,
                                                               batch_gold_for_loss_sub_task_tensor)
        else:
            raise Exception("Choose loss error !")

        one_batch_relation_res = (batch_gold_res_list, batch_pred_res_list)

        return one_batch_relation_res, one_batch_relation_loss
