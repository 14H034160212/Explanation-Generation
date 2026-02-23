import pandas as pd
import json
import random

path_list = ["./Paul_new_data/"]
all_test_set = pd.read_json(path_list[0]+"Sydney_all_generator_test_avg_3_lenexp_10.json")
all_train_set = pd.read_json(path_list[0]+"Sydney_all_generator_train_avg_3_lenexp_10.json")
first_round_set = pd.read_json(path_list[0]+"/Sydney/Sydney_vicuna_13b_finetuned_random_100.json")
# fine_tuned_vicuna_13b = pd.read_json(path_list[0]+"Vicuna_13B_Sydney_generator__avg_3_lenexp_10_Sydney_all_test_question_generated_explanation_bleu_bert_score_no_empty_explanation_no_s.json")
# vicuna_13b = pd.read_json(path_list[0]+"Vicuna_13B_Sydney_generator_Sydney_all_test_question_generated_explanation_bleu_bert_score_no_empty_explanation_no_s.json")

# 过滤all_train_set中的数据
filtered_test_set = all_test_set[~all_test_set['input'].isin(all_train_set['input'])]

# 过滤first_round_set中的数据
filtered_test_set = filtered_test_set[~filtered_test_set['input'].isin(first_round_set['input'])]

# 随机选择100条数据
random_samples = filtered_test_set.sample(100)

# 将数据写入json文件
# random_samples.to_json(path_list[0]+'Sydney/round2/Sydney_all_generator_test_avg_3_lenexp_10_round2_random_sample_100.json', orient='records')
data_dict = random_samples.to_dict('records')
with open(path_list[0]+'Sydney/round2/Sydney_all_generator_test_avg_3_lenexp_10_round2_random_sample_100.json', 'w') as f:
    json.dump(data_dict, f, indent=4)
