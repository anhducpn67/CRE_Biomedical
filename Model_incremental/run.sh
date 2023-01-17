

# single head
#  nohup python run.py --ID 7777 --GPU 0 --Training_way "Multi_Task_continual_Training" --Group_num 40 --Corpus_list "Twi_ADE" "ADE" "DDI" "CPR"  --All_data  >/dev/null 2>&1 &

# multi head
  #Multi-task continual
#  nohup python run.py --ID 7777 --GPU 1 --Training_way "Multi_Task_continual_Training" --Group_num 40 --Corpus_list "DDI" "Twi_ADE" "ADE"  --All_data  >/dev/null 2>&1 &
#  nohup python run.py --ID 7777 --GPU 3 --Training_way "Multi_Task_continual_Training" --Group_num 40 --Corpus_list "Twi_ADE" "ADE" "DDI" "CPR"  --All_data  >/dev/null 2>&1 &
  # Continual
#  nohup python run.py --ID 7777 --GPU 1 --Training_way "Multi_Task_continual_Training" --Group_num 1 --Corpus_list "DDI" "CPR" "Twi_ADE" "ADE"  --All_data  >/dev/null 2>&1 &



nohup python run.py --ID 7777 --GPU 0 --Training_way "Multi_Task_continual_Training" --Group_num 1 --Corpus_list "Twi_ADE" "ADE" "DDI" "CPR"  --All_data  >/dev/null 2>&1 &
