import json
import os

prefix_add = "77777"
task = "Combine_ADE"


def change_id(path_in, path_out):
    f = open(path_in, "r").readlines()
    f_out = open(path_out, "w")
    for line in f:
        obj = eval(line)
        obj["ID"] = prefix_add + obj["ID"]
        f_out.write(json.dumps(obj))
        f_out.write("\n")


files_path = [(f"../BIOES/base/{task}_{i}_base_model_data.json", f"../BIOES/base/new_{task}_{i}_base_model_data.json")
              for i in ["train", "valid", "test"]]
files_path += [(f"../BIOES/base/test/{task}_{i}_base_model_data.json", f"../BIOES/base/test/new_{task}_{i}_base_model_data.json")
               for i in ["train", "valid", "test"]]

for path_in, path_out in files_path:
    change_id(path_in, path_out)

[os.rename(f"../BIOES/base/{task}_{i}_base_model_data.json", f"../BIOES/base/old_{task}_{i}_base_model_data.json") for i in ["train", "valid", "test"]]
[os.rename(f"../BIOES/base/new_{task}_{i}_base_model_data.json", f"../BIOES/base/{task}_{i}_base_model_data.json") for i in ["train", "valid", "test"]]

[os.rename(f"../BIOES/base/test/{task}_{i}_base_model_data.json", f"../BIOES/base/test/old_{task}_{i}_base_model_data.json") for i in ["train", "valid", "test"]]
[os.rename(f"../BIOES/base/test/new_{task}_{i}_base_model_data.json", f"../BIOES/base/test/{task}_{i}_base_model_data.json") for i in ["train", "valid", "test"]]
