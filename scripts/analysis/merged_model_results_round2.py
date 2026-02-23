import pandas as pd
import json
import random

path_list = ["./Paul_new_data/"]
fine_tuned_vicuna_13b = pd.read_json(path_list[0]+"Vicuna_13B_Sydney_generator__avg_3_lenexp_10_Sydney_all_test_question_generated_explanation_bleu_bert_score_no_empty_explanation_no_s.json")
vicuna_13b = pd.read_json(path_list[0]+"Vicuna_13B_Sydney_generator_Sydney_all_test_question_generated_explanation_bleu_bert_score_no_empty_explanation_no_s.json")
gpt4 = pd.read_json(path_list[0]+"Sydney/round2/Sydney_gpt4_round2_random_100.json")


# 我们可以使用merge方法来找到input字段匹配的数据
merge1 = pd.merge(gpt4, fine_tuned_vicuna_13b, on='input', how='inner')
final_df = pd.merge(merge1, vicuna_13b, on='input', how='inner')

# 现在我们可以创建一个新的DataFrame来存放合并后的数据
merged_data = []

for index, row in final_df.iterrows():
    data = {}
    data['Question Stem'] = row['input'].split('Option A:')[0].strip()
    data['Answer options'] = 'Option A:' + row['input'].split('Option A:')[1].strip()
    data['Explanation 1'] = row['Explanation_x']
    data['Explanation 2'] = row['Generated_Explanation']
    data['Explanation 3'] = row['Generated_Explanation_y']
    data['Explanation 4'] = row['Generated_Explanation_x']
    merged_data.append(data)

merged_df = pd.DataFrame(merged_data)


# 将DataFrame转换为字典列表
merged_data_dict = merged_df.to_dict('records')

# 将字典列表保存为JSON文件
with open('/data/qbao775/Explanation-Generation/Paul_new_data/Sydney/round2/Sydney_Paul_round2_random_sample_100_data.json', 'w') as f:
    json.dump(merged_data_dict, f, indent=4)