import os
import numpy as np
import re

def process_0(raw_data, res_file):
    with open(raw_data, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()

    combined_data = []
    for index, item in enumerate(raw_data):
        data_item_list = item[:-1].split('|')
        assert len(data_item_list)==8
        new_data_item_list = []
        new_data_item_list.append(data_item_list[0])
        new_data_item_list.append(data_item_list[1])
        new_data_item_list.append(str((data_item_list[2:])))

        if index==0:
            combined_data.append(new_data_item_list)
        else:
            if raw_data[index].split('|')[1] == raw_data[index-1].split('|')[1]:
                temp = combined_data[-1]
                del combined_data[-1]
                new_data_item_list[2] = new_data_item_list[2] +"^"+ temp[2]
                combined_data.append(new_data_item_list)
            else:
                combined_data.append(new_data_item_list)

    with open(res_file, 'w') as f:
        for i in combined_data:
            new_data = "|".join(i)+"\n"
            f.writelines(new_data)
    return  res_file


def convert_docuemnt_sent_anno(anno, sent, temp_count):
    e1 = anno[0]
    e2 = anno[3]
    e1_raw_span = anno[1]
    e2_raw_span = anno[4]
    interval = abs(int(e1_raw_span)-int(e2_raw_span))
    E1_span, temp_count = return_sentence_pos(sent, e1, temp_count)
    E2_span, temp_count = return_sentence_pos(sent, e2, temp_count)

    if len(E1_span)==1 and len(E2_span)==1:
        return E1_span[0][0], E1_span[0][1], E2_span[0][0], E2_span[0][1], temp_count

    elif len(E1_span)>1 and len(E2_span)==1:
        for i in E1_span:
            if abs(i[0]-E2_span[0][0])==interval:
                return i[0], i[1], E2_span[0][0], E2_span[0][1], temp_count


    elif len(E2_span)>1 and len(E1_span)==1:
        for i in E2_span:
            if abs(i[0]-E1_span[0][0])==interval:
                return E1_span[0][0], E1_span[0][1], i[0], i[1], temp_count


    elif len(E2_span)>1 and len(E1_span)>1:
        for e1 in E1_span:
            for e2 in E2_span:
                if abs(e1[0]-e2[0])==interval:
                    return e1[0], e1[1], e2[0], e2[1], temp_count

    else:
        raise Exception("convert_docuemnt_sent_anno")
    return 0, 0, 0, 0, temp_count




def return_sentence_pos(sentence, text, temp_count):
    sentence_start_end_list = []

    try:
        new_text = re.sub("\(", "\\(", text)
        new_text = re.sub("\)", "\\)", new_text)
        new_text = re.sub("\?", "\\?", new_text)
        new_text = re.sub("\+", "\\+", new_text)
        new_text = re.sub("\$", "\\$", new_text)
        new_text = re.sub("\[", "\\[", new_text)
        new_text = re.sub("\]", "\\]", new_text)
        new_text = re.sub("\*", "\\*", new_text)
        new_text = re.sub("\.", "\\.", new_text)

        # for m in re.finditer(sentence, text):
        #     sentence_start_end_list.append((m.start(), m.end()))

        temp_finditer = [i for i in re.finditer(new_text, sentence)]
        for i in temp_finditer:
            sentence_start_end_list.append(i.span())
            # if len(temp_finditer) == 1:
            #     sentence_start_end_list.append(temp_finditer[0].span())
            # elif len(temp_finditer) > 1:
            #     # only take the first one
            #     sentence_start_end_list.append(temp_finditer[0].span())
            #     temp_count += 1
            #     print(temp_count)
            #     # break
            #     print("more than one find !")
            # elif len(temp_finditer) == 0:
            #     print("no find !")
    except:
        print("re.finder error~~~~~~~~~~~~~~~~~~~~~~")

    return sentence_start_end_list, temp_count


def process_1(res_file):

    with open(res_file, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()

    test_valid_num = int( len(raw_data) *0.1 )


    train_index = list(range(len(raw_data)))
    valid_index = []
    test_index = []

    for i in range(test_valid_num):
        valid_randIndex = int(np.random.uniform(0, len(train_index) )) # 获得0~len(trainingSet)的一个随机数
        valid_index.append(train_index[valid_randIndex])
        del(train_index[valid_randIndex])

        test_randIndex = int(np.random.uniform(0, len(train_index) )) # 获得0~len(trainingSet)的一个随机数
        test_index.append(train_index[test_randIndex])
        del(train_index[test_randIndex])


    file_list = ["ADE_train.csv", "ADE_valid.csv", "ADE_test.csv"]
    index_list = [train_index, valid_index, test_index]
    temp_count = 0
    file_num = 0
    for index, file in enumerate(file_list):
        file = os.path.join('../data_csv', file)
        with open(file, 'w') as f:
            for data_index in index_list[index]:
                data = raw_data[data_index].split('|')
                data_anno = data[2].split("^")
                new_data_anno_list = []
                for anno in data_anno:
                    anno = eval(anno)
                    assert  len(anno)==6
                    # ('D003693', 'delirium', (1061, 1069), 'Disease')
                    # ['hemoptysis', '299', '309', 'acyclovir', '259', '268']
                    anno_S_1, anno_S_2, anno_E_1, anno_E_2, temp_count = convert_docuemnt_sent_anno(anno, data[1], temp_count)

                    if anno_S_1 == 0 and  anno_S_2== 0 and anno_E_1== 0 and anno_E_2 == 0:
                        print("no found", data_index)

                    anno_S = ('entity_ID', anno[0], (anno_S_1, anno_S_2), "adverse_effect")
                    anno_E = ('entity_ID',anno[3], (anno_E_1, anno_E_2), "drug")
                    if anno_S_1<anno_E_1:
                        new_anno = [anno_S, anno_E, 'ADE']
                    else:
                        new_anno = [anno_E, anno_S, 'ADE']
                    new_data_anno_list.append(new_anno)

                data[0] = str(file_num)
                file_num+=1
                data[2] = str(new_data_anno_list)
                f.writelines("||".join(data)+"\n")

if __name__ == '__main__':
    raw_data = '../raw_data/DRUG-AE.rel'
    res_file = '../data_csv/ADE.csv'

    # res_file = process_0(raw_data, res_file)

    process_1(res_file)


