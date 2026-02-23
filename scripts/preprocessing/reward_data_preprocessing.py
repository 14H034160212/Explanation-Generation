import pandas as pd
import json
import random

path_list = ["./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/"]
gpt4_sampled = pd.read_json(path_list[0]+"GPT-4_generator_merged_avg_3_lenexp_10_sample_5000.json")
llama2_13b_sampled = pd.read_json(path_list[0]+"Llama_2_13B_all_generator_sample_5000_question_generated_explanation.json")
vicuna_13b_sampled = pd.read_json(path_list[0]+"Vicuna_13B_all_generator_sample_5000_question_generated_explanation.json")

df = pd.DataFrame(columns=["prompt", "chosen", "rejected"])
for index, row in gpt4_sampled.iterrows():
    data = {}
    data['prompt'] = row['input']
    data['chosen'] = row['Generated_Explanation']
    data['rejected'] = llama2_13b_sampled.loc[index]['Generated_Explanation']
    df = df.append(data, ignore_index=True)
    
    data2 = {}
    data2['prompt'] = row['input']
    data2['chosen'] = row['Generated_Explanation']
    data2['rejected'] = vicuna_13b_sampled.loc[index]['Generated_Explanation']
    df = df.append(data2, ignore_index=True)

# 把df保存到json文件中
df.to_json(path_list[0]+"GPT-4_Llama_2_Vicuna_13B_reward_generator_sample_5000_question_generated_explanation.json", orient='records', indent=4)