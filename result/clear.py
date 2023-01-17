
import os
import shutil

def del_file(path_data):
    for i in os.listdir(path_data):
        path_file = os.path.join(path_data, i)
        try:
            shutil.rmtree(path_file)
        except:
            os.remove(path_file)



path_data = r"./detail_performance"
del_file(path_data)
path_data = r"./detail_results"
del_file(path_data)
path_data = r"./detail_training"
del_file(path_data)
path_data = r"./save_model"
del_file(path_data)
path_data = r"./runs"
del_file(path_data)
