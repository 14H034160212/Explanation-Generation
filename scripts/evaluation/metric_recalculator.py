import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score
import json

input_data = pd.read_json("./Paul_new_data/Sydney/Sydney_gpt-4_random_100.json")
for index, row in input_data.iterrows():
    precision, recall, bertscore = score([row["Generated_Explanation"]], [row["Explanation"]], lang="en", model_type="bert-base-uncased", verbose=False)
    bertscore = bertscore.item()
    
    # 为新的列设置值
    input_data.at[index, 'bleu_score'] = sentence_bleu([row["Explanation"]], row["Generated_Explanation"])
    input_data.at[index, 'bert_score'] = bertscore
    
    
# 将DataFrame转换为字典列表
merged_data_dict = input_data.to_dict('records')

# 将字典列表保存为JSON文件
with open('./Paul_new_data/Sydney/Sydney_gpt-4_random_100_2.json', 'w') as f:
    json.dump(merged_data_dict, f, indent=4)
    
input_data.to_excel('./Paul_new_data/Sydney/Sydney_gpt-4_random_100_2.xlsx', index=False)
