import torch.nn as nn
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class MyBinaryClassifier(nn.Module):
    def __init__(self, TAGS_fields, input_dim, device, ignore_index=None, loss_weight=None):
        super(MyBinaryClassifier, self).__init__()
        self.to(device)
        self.ignore_index = ignore_index
        self.input_dim = input_dim
        self.output_dim = len(TAGS_fields.vocab)
        self.fc1 = nn.Linear(self.input_dim, int(self.input_dim / 2), bias=False)
        self.fc2 = nn.Linear(int(self.input_dim / 2), self.output_dim, bias=False)
        self.loss_weight = loss_weight

    def forward(self, common_embedding):
        # common_embedding = nn.Dropout(p=0.5)(common_embedding)
        res = F.relu(self.fc1(common_embedding))
        res = self.fc2(res)
        return res

    def get_ce_loss(self, res, targets):
        cross_entropy_loss = torch.nn.functional.cross_entropy(res.permute(0, 2, 1), targets,
                                                               ignore_index=self.ignore_index, weight=self.loss_weight)
        return cross_entropy_loss


class MyRelationClassifier(nn.Module):
    def __init__(self, args, device):
        super(MyRelationClassifier, self).__init__()
        self.to(device)
        self.args = args
        self.device = device
        self.TAGS_my_types_classification = torchtext.legacy.data.Field(dtype=torch.long, batch_first=True,
                                                                        pad_token=None, unk_token=None)
        self.TAGS_my_types_classification.vocab = {"no": 0, "yes": 1}
        self.ignore_index = len(self.TAGS_my_types_classification.vocab)
        if self.args.Entity_Prep_Way == "entity_type_marker":
            self.relation_input_dim = self.args.Word_embedding_size
        else:
            self.relation_input_dim = 2 * self.args.Word_embedding_size
        if self.args.Weight_Loss:
            self.yes_no_weight = torch.tensor([self.args.Min_weight, self.args.Max_weight], device=self.device)
            print("self.yes_no_weight", self.yes_no_weight)
        else:
            self.yes_no_weight = None

    def get_binary_classifier(self, i):
        return getattr(self, 'my_classifier_{0}'.format(i))

    def create_classifiers(self, TAGS_Types_fields_dic,
                           TAGS_sep_entity_fields_dic,
                           TAGS_Entity_Type_fields_dic):
        self.TAGS_Types_fields_dic = TAGS_Types_fields_dic
        self.TAGS_Entity_Type_fields_dic = TAGS_Entity_Type_fields_dic
        self.TAGS_sep_entity_fields_dic = TAGS_sep_entity_fields_dic
        self.relation_list = list(self.TAGS_Types_fields_dic.keys())
        self.entity_type_list = list(self.TAGS_Entity_Type_fields_dic.keys())

        for relation in self.relation_list:
            my_binary_classifier = MyBinaryClassifier(self.TAGS_my_types_classification,
                                                      self.relation_input_dim, self.device,
                                                      ignore_index=self.ignore_index)
            setattr(self, f'my_classifier_{relation}', my_binary_classifier)

    def forward(self, batch_added_marker_entity_span_vec):
        loss_list = []
        sub_task_res_prob_list = []
        sub_task_res_yes_no_index_list = []
        for relation in self.relation_list:
            res = self.get_binary_classifier(relation)(batch_added_marker_entity_span_vec)
            loss_list.append(res)
            res = torch.max(res, 2)
            sub_task_res_prob_list.append(res[0])
            sub_task_res_yes_no_index_list.append(res[1])

        batch_pred_res_prob = torch.stack(sub_task_res_prob_list).permute(1, 2, 0)
        batch_pred_res_yes_no_index = torch.stack(sub_task_res_yes_no_index_list).permute(1, 2, 0)

        yes_flag_tensor = torch.tensor(self.TAGS_my_types_classification.vocab["yes"], device=self.device)
        no_mask_tensor = batch_pred_res_yes_no_index != yes_flag_tensor

        batch_pred_res_prob_masked = torch.masked_fill(batch_pred_res_prob, no_mask_tensor,
                                                       torch.tensor(-999, device=self.device))

        # deal the solution of all results are no, there is None classifier, commented down these two lines
        pad_tensor = torch.Tensor(batch_pred_res_prob_masked.shape[0], batch_pred_res_prob_masked.shape[1], 1).fill_(-998).to(self.device)
        batch_pred_res_prob_masked = torch.cat((batch_pred_res_prob_masked, pad_tensor), 2)

        pred_type_tensor = torch.max(batch_pred_res_prob_masked, 2)[1]

        return pred_type_tensor, loss_list

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
                pad_sub_task_list = [torch.tensor(self.ignore_index, dtype=torch.long, device=self.device)] * len(
                    self.relation_list)
                one_sent_list.append(torch.stack(pad_sub_task_list))

            batch_gold_for_loss_sub_task_list.append(torch.stack(one_sent_list))

        gold_for_loss_sub_task_tensor = pad_sequence(batch_gold_for_loss_sub_task_list, batch_first=True,
                                                     padding_value=self.ignore_index)
        gold_for_loss_sub_task_tensor = gold_for_loss_sub_task_tensor.permute(2, 0, 1)

        return gold_for_loss_sub_task_tensor

    def get_ensembled_ce_loss(self, batch_pred_for_loss_sub_task_list, batch_gold_for_loss_sub_task_tensor):
        loss_list = []
        for sub_task_index, sub_task_batch_pred in enumerate(batch_pred_for_loss_sub_task_list):
            sub_task_batch_gold = batch_gold_for_loss_sub_task_tensor[sub_task_index]
            ce_loss = F.cross_entropy(sub_task_batch_pred.permute(0, 2, 1), sub_task_batch_gold,
                                      ignore_index=self.ignore_index, weight=self.yes_no_weight, reduction='mean')
            loss_list.append(ce_loss)
        return sum(loss_list) / len(loss_list)

    def BCE_loss(self, batch_pred_for_loss_sub_task_list, batch_gold_for_loss_sub_task_tensor):
        loss_list = []
        for sub_task_index, sub_task_batch_pred in enumerate(batch_pred_for_loss_sub_task_list):
            sub_task_batch_gold = batch_gold_for_loss_sub_task_tensor[sub_task_index]
            target_tensor = torch.where(sub_task_batch_gold == self.ignore_index, torch.tensor(0, device=self.device), sub_task_batch_gold)
            target_tensor = torch.zeros(sub_task_batch_pred.shape, device=self.device).scatter_(2, target_tensor.unsqueeze(2), 1)

            criterion = nn.BCEWithLogitsLoss(pos_weight=self.yes_no_weight, reduction='mean')
            bec_loss = criterion(sub_task_batch_pred, target_tensor)
            loss_list.append(bec_loss)
        return sum(loss_list) / len(loss_list)
