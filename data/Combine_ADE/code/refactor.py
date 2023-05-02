import json
import os

task = "Combine_ADE"


def refactor(path_in, path_out):
    f = open(path_in, "r").readlines()
    f_out = open(path_out, "w")
    for line in f:
        obj = eval(line)
        del obj["entity_span"]
        del obj["entity_span_and_type"]
        del obj["joint_entity_type_Drug"]
        del obj["joint_entity_type_Disease"]
        del obj["sampled_entity_span"]
        obj["Drug"] = obj.pop("only_entity_type_Drug")
        obj["Disease"] = obj.pop("only_entity_type_Disease")
        obj["Drug_Disease_interaction"] = obj.pop("relation_Drug_Disease_interaction")
        f_out.write(json.dumps(obj))
        f_out.write("\n")


files_path = [(f"../BIOES/base/{task}_{i}_base_model_data.json", f"../BIOES/base/new_{task}_{i}_base_model_data.json")
              for i in ["train", "valid", "test"]]
files_path += [
    (f"../BIOES/base/test/{task}_{i}_base_model_data.json", f"../BIOES/base/test/new_{task}_{i}_base_model_data.json")
    for i in ["train", "valid", "test"]]

for path_in, path_out in files_path:
    refactor(path_in, path_out)

[os.rename(f"../BIOES/base/{task}_{i}_base_model_data.json", f"../BIOES/base/old_{task}_{i}_base_model_data.json") for i
 in ["train", "valid", "test"]]
[os.rename(f"../BIOES/base/new_{task}_{i}_base_model_data.json", f"../BIOES/base/{task}_{i}_base_model_data.json") for i
 in ["train", "valid", "test"]]

[os.rename(f"../BIOES/base/test/{task}_{i}_base_model_data.json",
           f"../BIOES/base/test/old_{task}_{i}_base_model_data.json") for i in ["train", "valid", "test"]]
[os.rename(f"../BIOES/base/test/new_{task}_{i}_base_model_data.json",
           f"../BIOES/base/test/{task}_{i}_base_model_data.json") for i in ["train", "valid", "test"]]
