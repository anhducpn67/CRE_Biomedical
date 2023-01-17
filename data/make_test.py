import json
import os
import shutil


dir_list = ["CPR","DDI","CDR","ADE","Twi_ADE"]


raw_file_list = ["BIOES/base/task_name_train_base_model_data.json", "BIOES/base/task_name_valid_base_model_data.json", "BIOES/base/task_name_test_base_model_data.json"]
new_file_list = ["BIOES/base/test/task_name_train_base_model_data.json", "BIOES/base/test/task_name_valid_base_model_data.json", "BIOES/base/test/task_name_test_base_model_data.json"]

test_num = 10

for dir in dir_list:
    for index, file in enumerate(raw_file_list):
        raw_file = os.path.join(dir, file.replace("task_name", dir))

        with open(raw_file, "r") as f:
            raw_data = f.readlines()
        print("file:{0}, len={1}".format(file, len(raw_data)))

        new_file = os.path.join(dir, new_file_list[index].replace("task_name", dir))
        with open(new_file, "w") as f:
            for i in raw_data[:test_num]:
                f.write(json.dumps(eval(i)))
                f.write("\n")

model_test_data_path = "Multi_Task_Training/base/test"
shutil.rmtree(model_test_data_path)
os.makedirs("./Multi_Task_Training/base/test")



raw_file_list = ["BIOES/large/task_name_train_large_model_data.json", "BIOES/large/task_name_valid_large_model_data.json", "BIOES/large/task_name_test_large_model_data.json"]
new_file_list = ["BIOES/large/test/task_name_train_large_model_data.json", "BIOES/large/test/task_name_valid_large_model_data.json", "BIOES/large/test/task_name_test_large_model_data.json"]


for dir in dir_list:
    for index, file in enumerate(raw_file_list):
        raw_file = os.path.join(dir, file.replace("task_name", dir))

        with open(raw_file, "r") as f:
            raw_data = f.readlines()
        print("file:{0}, len={1}".format(file, len(raw_data)))

        new_file = os.path.join(dir, new_file_list[index].replace("task_name", dir))
        with open(new_file, "w") as f:
            for i in raw_data[:test_num]:
                f.write(json.dumps(eval(i)))
                f.write("\n")

model_test_data_path = "Multi_Task_Training/large/test"
shutil.rmtree(model_test_data_path)
os.makedirs("./Multi_Task_Training/large/test")