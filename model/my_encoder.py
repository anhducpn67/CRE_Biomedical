import torch.nn as nn
import torch
from itertools import combinations
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import copy


class MyEncoder(nn.Module):
    def __init__(self, bert, tokenizer, args, device):
        super(MyEncoder, self).__init__()
        self.to(device)
        self.args = args
        self.device = device
        self.bert = bert
        self.tokenizer = tokenizer
        self.linear_transform = nn.Linear(self.args.Word_embedding_size * 2, self.args.Word_embedding_size)
        self.layer_normalization = nn.LayerNorm([self.args.Word_embedding_size])
        self.output_dim = args.Word_embedding_size

    def forward(self, batch_inputs, ignore_index=0, encoder_hidden_states=None):
        tokens_tensor = self.get_bert_input(batch_inputs)
        attention_mask = self.get_attention_mask(tokens_tensor, ignore_index)
        position_ids = self.get_position_ids(tokens_tensor)
        common_embedding = self.bert(tokens_tensor, output_hidden_states=True, attention_mask=attention_mask,
                                     position_ids=position_ids, encoder_hidden_states=encoder_hidden_states)

        last_common_embedding = common_embedding[0]

        return last_common_embedding

    def get_bert_input(self, batch_inputs):
        input_tensor = torch.tensor(batch_inputs, device=self.device)
        return input_tensor

    def get_attention_mask(self, tokens_tensor, ignore_index):
        attention_mask = torch.where(tokens_tensor == ignore_index, torch.tensor(0., device=self.device),
                                     torch.tensor(1., device=self.device))
        return attention_mask

    def get_position_ids(self, tokens_tensor):
        position_ids = torch.arange(tokens_tensor.shape[1], device=self.device).expand((1, -1))
        return position_ids

    def get_entity_pair_rep(self, entity_pair_span, sentence_embedding):
        entity_span_1, entity_span_2 = entity_pair_span
        entity_1_head = entity_span_1[0]
        entity_1_tail = entity_span_1[-1]
        entity_2_head = entity_span_2[0]
        entity_2_tail = entity_span_2[-1]

        if self.args.Entity_Prep_Way == "entity_type_marker":
            entity_1 = sentence_embedding[entity_1_head]
            entity_2 = sentence_embedding[entity_2_head]
        elif self.args.Entity_Prep_Way == "standard":
            entity_1 = torch.sum(sentence_embedding[entity_1_head:entity_1_tail], dim=0).squeeze()
            entity_2 = torch.sum(sentence_embedding[entity_2_head:entity_2_tail], dim=0).squeeze()
        else:
            raise Exception("args.Entity_Prep_Way error !")

        entity_pair_rep = torch.cat((entity_1, entity_2))
        return entity_pair_rep

    def batch_get_entity_pair_rep(self, batch_tokens, batch_entity, batch_entity_type):
        padding_value = self.tokenizer.vocab['[PAD]']

        if self.args.Entity_Prep_Way == "entity_type_marker":
            raw_batch_entity = copy.deepcopy(batch_entity)
            batch_tokens_marker = []
            for sent_index, (one_sent_tokens, one_sent_entity, one_sent_entity_type) in enumerate(
                    zip(batch_tokens, batch_entity, batch_entity_type)):
                temp_token_list = one_sent_tokens.tolist()
                for entity_index, (span, entity_type) in enumerate(zip(one_sent_entity, one_sent_entity_type)):
                    temp_token_list.insert(span[0], self.tokenizer.convert_tokens_to_ids("[Entity_" + entity_type + "]"))
                    temp_token_list.insert(span[-1] + 2, self.tokenizer.convert_tokens_to_ids("[/Entity_" + entity_type + "]"))

                    batch_entity[sent_index][entity_index].insert(0, span[0])
                    batch_entity[sent_index][entity_index].append(span[-1] + 2)

                    # need batch_span is in order in entity list
                    batch_entity[sent_index][entity_index + 1:] = [[j + 2 for j in i] for i in
                                                                   batch_entity[sent_index][entity_index + 1:]]
                    batch_entity[sent_index][entity_index][1:-1] = [i + 1 for i in
                                                                    batch_entity[sent_index][entity_index][1:-1]]
                batch_tokens_marker.append(torch.tensor(temp_token_list, device=self.device))

            common_embedding = self.forward(
                pad_sequence(batch_tokens_marker, batch_first=True, padding_value=padding_value),
                encoder_hidden_states=None)

            batch_entity_pair_span_list = []
            batch_entity_pair_vec_list = []
            batch_sent_len_list = []

            for sent_index, (raw_one_sent_entity, one_sent_entity, one_sent_entity_type) in enumerate(
                    zip(raw_batch_entity, batch_entity, batch_entity_type)):
                sent_entity_pair_span_list = list(combinations(one_sent_entity, 2))
                sent_entity_pair_span_list = [sorted(i) for i in sent_entity_pair_span_list]
                batch_sent_len_list.append(len(sent_entity_pair_span_list))

                raw_sent_entity_pair_span_list = list(combinations(raw_one_sent_entity, 2))
                raw_sent_entity_pair_span_list = [sorted(i) for i in raw_sent_entity_pair_span_list]
                batch_entity_pair_span_list.append(raw_sent_entity_pair_span_list)

                sent_entity_pair_rep_list = []
                if sent_entity_pair_span_list:
                    for entity_pair_span in sent_entity_pair_span_list:
                        one_sent_rep = self.get_entity_pair_rep(entity_pair_span, common_embedding[sent_index])
                        sent_entity_pair_rep_list.append(one_sent_rep)
                else:
                    sent_entity_pair_rep_list.append(torch.tensor([0] * self.output_dim * 2).float().to(self.device))

                batch_entity_pair_vec_list.append(torch.stack(sent_entity_pair_rep_list))

            batch_added_marker_entity_span_vec = pad_sequence(batch_entity_pair_vec_list,
                                                              batch_first=True, padding_value=padding_value)

            batch_added_marker_entity_span_vec = self.linear_transform(batch_added_marker_entity_span_vec)
            batch_added_marker_entity_span_vec = F.gelu(batch_added_marker_entity_span_vec)
            batch_added_marker_entity_span_vec = self.layer_normalization(batch_added_marker_entity_span_vec)

            return batch_added_marker_entity_span_vec, batch_entity_pair_span_list, batch_sent_len_list

        elif self.args.Entity_Prep_Way == "standard":
            common_embedding = self.forward(batch_tokens, encoder_hidden_states=None, ignore_index=padding_value)
            batch_entity_pair_span_list = []
            batch_entity_pair_vec_list = []
            batch_sent_len_list = []

            for sent_index, (one_sent_entity, one_sent_entity_type) in enumerate(zip(batch_entity, batch_entity_type)):
                sent_entity_pair_span_list = list(combinations(one_sent_entity, 2))
                batch_sent_len_list.append(len(sent_entity_pair_span_list))
                sent_entity_pair_span_list = [sorted(i) for i in sent_entity_pair_span_list]

                sent_entity_pair_rep_list = []
                if sent_entity_pair_span_list:
                    for entity_pair_span in sent_entity_pair_span_list:
                        one_sent_rep = self.get_entity_pair_rep(entity_pair_span, common_embedding[sent_index])
                        sent_entity_pair_rep_list.append(one_sent_rep)
                else:
                    sent_entity_pair_rep_list.append(
                        torch.tensor([0] * self.relation_input_dim).float().to(self.device))

                batch_entity_pair_vec_list.append(torch.stack(sent_entity_pair_rep_list))
                batch_entity_pair_span_list.append(sent_entity_pair_span_list)
            batch_added_marker_entity_span_vec = pad_sequence(batch_entity_pair_vec_list,
                                                              batch_first=True, padding_value=padding_value)

            return batch_added_marker_entity_span_vec, batch_entity_pair_span_list, batch_sent_len_list
        else:
            raise Exception("Entity_Prep_Way wrong !")

    def memory_get_entity_pair_rep(self, batch_entity, batch_tokens, batch_entity_type, batch_gold_RE):
        padding_value = self.tokenizer.vocab['[PAD]']

        raw_batch_entity = copy.deepcopy(batch_entity)
        batch_tokens_marker = []
        for sent_index, (one_sent_tokens, one_sent_entity, one_sent_type) in enumerate(
                zip(batch_tokens, batch_entity, batch_entity_type)):
            temp_token_list = one_sent_tokens.tolist()
            for entity_index, (span, entity_type) in enumerate(zip(one_sent_entity, one_sent_type)):
                temp_token_list.insert(span[0], self.tokenizer.convert_tokens_to_ids("[Entity_" + entity_type + "]"))
                temp_token_list.insert(span[-1] + 2,
                                       self.tokenizer.convert_tokens_to_ids("[/Entity_" + entity_type + "]"))

                batch_entity[sent_index][entity_index].insert(0, span[0])
                batch_entity[sent_index][entity_index].append(span[-1] + 2)

                # need batch_span is in order in entity list
                batch_entity[sent_index][entity_index + 1:] = [[j + 2 for j in i] for i in
                                                               batch_entity[sent_index][entity_index + 1:]]
                batch_entity[sent_index][entity_index][1:-1] = [i + 1 for i in
                                                                batch_entity[sent_index][entity_index][1:-1]]
            batch_tokens_marker.append(torch.tensor(temp_token_list, device=self.device))

        common_embedding = self.forward(pad_sequence(batch_tokens_marker, batch_first=True, padding_value=padding_value))

        batch_entity_pair_span_list = []
        batch_entity_pair_vec_list = []
        batch_sent_len_list = []

        for sent_index, (raw_one_sent_entity, one_sent_entity, one_sent_type) in enumerate(
                zip(raw_batch_entity, batch_entity, batch_entity_type)):
            temp_sent_entity_pair_span_list = list(combinations(one_sent_entity, 2))
            temp_sent_entity_pair_span_list = [sorted(i) for i in temp_sent_entity_pair_span_list]

            temp_raw_sent_entity_pair_span_list = list(combinations(raw_one_sent_entity, 2))
            temp_raw_sent_entity_pair_span_list = [sorted(i) for i in temp_raw_sent_entity_pair_span_list]

            sent_entity_pair_span_list = []
            raw_sent_entity_pair_span_list = []
            for idx, entity_pair_span in enumerate(temp_raw_sent_entity_pair_span_list):
                if entity_pair_span in batch_gold_RE[sent_index]:
                    sent_entity_pair_span_list.append(temp_sent_entity_pair_span_list[idx])
                    raw_sent_entity_pair_span_list.append(temp_raw_sent_entity_pair_span_list[idx])
            batch_sent_len_list.append(len(sent_entity_pair_span_list))

            batch_entity_pair_span_list.append(raw_sent_entity_pair_span_list)

            sent_entity_pair_rep_list = []
            if sent_entity_pair_span_list:
                for entity_pair_span in sent_entity_pair_span_list:
                    one_sent_rep = self.get_entity_pair_rep(entity_pair_span, common_embedding[sent_index])
                    sent_entity_pair_rep_list.append(one_sent_rep)
            else:
                sent_entity_pair_rep_list.append(
                    torch.tensor([0] * self.output_dim * 2).float().to(self.device))

            batch_entity_pair_vec_list.append(torch.stack(sent_entity_pair_rep_list))

        batch_added_marker_entity_span_vec = pad_sequence(batch_entity_pair_vec_list, batch_first=True,
                                                          padding_value=padding_value)

        batch_added_marker_entity_span_vec = self.linear_transform(batch_added_marker_entity_span_vec)
        batch_added_marker_entity_span_vec = F.gelu(batch_added_marker_entity_span_vec)
        batch_added_marker_entity_span_vec = self.layer_normalization(batch_added_marker_entity_span_vec)

        return batch_added_marker_entity_span_vec, batch_entity_pair_span_list, batch_sent_len_list
