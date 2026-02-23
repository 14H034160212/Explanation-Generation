import pandas as pd
from sklearn.model_selection import train_test_split
import json

datasets = {
    "sydney": pd.read_json("./Paul_new_data/Sydney/Sydney_merged_verifier_way_2.json"),
    "cardiff": pd.read_json("./Paul_new_data/Cardiff/Cardiff_merged_verifier_way_2.json"),
    "auckland_law": pd.read_json("./PeerWiseData/Law/Auckland_law_merged_verifier_way_2.json"),
    "uk_medical_year1": pd.read_json("./PeerWiseData/Medicine/UK_medicine_year1_merged_verifier_way_2.json"),
    "uk_medical_year2": pd.read_json("./PeerWiseData/Medicine/UK_medicine_year2_merged_verifier_way_2.json")
}

test_sets = []
train_sets = []

name_list = ['Sydney', 'Cardiff', 'Law', 'Medicine', 'Medicine']
flag = 0
for name, df in datasets.items():
    df.drop_duplicates(subset='input', keep='first', inplace=True)  # 删除重复数据
    train, test = train_test_split(df, test_size=100, random_state=42)  # 使用sklearn的train_test_split方法，随机分割数据集
    test_sets.append(test)
    train_sets.append(train)

all_test_inputs = set()
for test in test_sets:
    all_test_inputs.update(test['input'])

# Remove all test set inputs from each train set
for i in range(len(train_sets)):
    train_sets[i] = train_sets[i][~train_sets[i]['input'].isin(all_test_inputs)]

# Save to JSON after removing overlaps
for i, name in enumerate(datasets.keys()):
    test = test_sets[i]
    train = train_sets[i]
    
    # 将每个测试集和训练集分别保存为JSON文件
    if i <= 1:
        test.to_json(f'./Paul_new_data/{name_list[i]}/evaluator_Test_{name}.json', orient='records')
        train.to_json(f'./Paul_new_data/{name_list[i]}/evaluator_Train_{name}.json', orient='records')
    else:
        test.to_json(f'./PeerWiseData/{name_list[i]}/evaluator_Test_{name}.json', orient='records')
        train.to_json(f'./PeerWiseData/{name_list[i]}/evaluator_Train_{name}.json', orient='records')        

# 合并所有训练集为一个大的训练集
all_train = pd.concat(train_sets)

# 将DataFrame转换为字典列表
merged_data_dict = all_train.to_dict('records')

# 将字典列表保存为JSON文件
with open('./Paul_new_data/evaluator_Train_all.json', 'w') as f:
    json.dump(merged_data_dict, f, indent=4)

# train_data.to_json('./Paul_new_data/evaluator_Train_all.json', orient='records')

# 保存合并后的训练集为JSON文件
# all_train.to_json('./Paul_new_data/evaluator_Train_all.json', orient='records')
