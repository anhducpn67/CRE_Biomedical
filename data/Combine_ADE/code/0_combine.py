import random


def combine():
    f1 = open(file1, "r").readlines()
    f2 = open(file2, "r").readlines()
    f_combine = f1 + f2
    random.shuffle(f_combine)
    f = open(combine_file, "w")
    for sent in f_combine:
        f.write(sent)


types = ["train", "valid", "test"]

for t in types:
    file1 = f"../raw_data/ADE_{t}_base_model_data.json"
    file2 = f"../raw_data/Twi_ADE_{t}_base_model_data.json"
    combine_file = f"../BIOES/base/Combine_ADE_{t}_base_model_data.json"
    combine()
