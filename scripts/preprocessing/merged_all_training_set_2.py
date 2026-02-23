import pandas as pd
import json

sydney_training = pd.read_json("./Paul_new_data/Sydney/Sydney_all_generator_train_avg_3_lenexp_10.json")
cardiff_training = pd.read_json("./Paul_new_data/Cardiff/Cardiff_all_generator_train_avg_3_lenexp_10.json")
auckland_law_training = pd.read_json("./PeerWiseData/Law/Auckland_law_all_generator_train_avg_3_lenexp_10.json")
uk_medical_year1_training = pd.read_json("./PeerWiseData/Medicine/UK_medical_year1_all_generator_train_avg_3_lenexp_10.json")
uk_medical_year2_training = pd.read_json("./PeerWiseData/Medicine/UK_medical_year2_all_generator_train_avg_3_lenexp_10.json")


sydney_test = pd.read_json("./Paul_new_data/Sydney/Sydney_all_generator_test_avg_3_lenexp_10.json")
cardiff_test = pd.read_json("./Paul_new_data/Cardiff/Cardiff_all_generator_test_avg_3_lenexp_10.json")
auckland_law_test = pd.read_json("./PeerWiseData/Law/Auckland_law_all_generator_test_avg_3_lenexp_10.json")
uk_medical_year1_test = pd.read_json("./PeerWiseData/Medicine/UK_medical_year1_all_generator_test_avg_3_lenexp_10.json")
uk_medical_year2_test = pd.read_json("./PeerWiseData/Medicine/UK_medical_year2_all_generator_test_avg_3_lenexp_10.json")


sydney_test_100 = pd.read_json("./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json")
cardiff_test_100 = pd.read_json("./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json")
auckland_law_test_100 = pd.read_json("./PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json")
uk_medical_year1_test_100 = pd.read_json("./PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json")
uk_medical_year2_test_100 = pd.read_json("./PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json")



test_sets_100 = [sydney_test_100, cardiff_test_100, auckland_law_test_100, uk_medical_year1_test_100, uk_medical_year2_test_100]
test_100 = pd.concat(test_sets_100)


# 合并所有的训练集
train_sets = [sydney_training, cardiff_training, auckland_law_training, uk_medical_year1_training, uk_medical_year2_training,sydney_test, cardiff_test, auckland_law_test, uk_medical_year1_test, uk_medical_year2_test]
train = pd.concat(train_sets)



all_test_inputs = set()
for tes in test_sets_100:
    all_test_inputs.update(tes['input'])
    
# 去除训练集中与测试集相同的部分
for i in range(len(train_sets)):
    train_sets[i] = train_sets[i][~train_sets[i]['input'].isin(all_test_inputs)]


all_train = pd.concat(train_sets)

# 将DataFrame转换为字典列表
merged_training_dict = all_train.to_dict('records')
# 将字典列表保存为JSON文件
with open('./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10_update.json', 'w') as f:
    json.dump(merged_training_dict, f, indent=4)


# 将DataFrame转换为字典列表
sydney_training_dict = train_sets[0].to_dict('records')
# 将字典列表保存为JSON文件
with open('./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/Sydney_all_generator_train_avg_3_lenexp_10_update.json', 'w') as f:
    json.dump(sydney_training_dict, f, indent=4)
    
    
# 将DataFrame转换为字典列表
cardiff_training_dict = train_sets[1].to_dict('records')
# 将字典列表保存为JSON文件
with open('./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/Cardiff_all_generator_train_avg_3_lenexp_10_update.json', 'w') as f:
    json.dump(cardiff_training_dict, f, indent=4)
    
    
# 将DataFrame转换为字典列表
auckland_law_training_dict = train_sets[2].to_dict('records')
# 将字典列表保存为JSON文件
with open('./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/Auckland_law_all_generator_train_avg_3_lenexp_10_update.json', 'w') as f:
    json.dump(auckland_law_training_dict, f, indent=4)
    

# 将DataFrame转换为字典列表
uk_medical_year1_training_dict = train_sets[3].to_dict('records')
# 将字典列表保存为JSON文件
with open('./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/UK_medical_year1_all_generator_train_avg_3_lenexp_10_update.json', 'w') as f:
    json.dump(uk_medical_year1_training_dict, f, indent=4)
    
    
# 将DataFrame转换为字典列表
uk_medical_year2_training_dict = train_sets[4].to_dict('records')
# 将字典列表保存为JSON文件
with open('./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/UK_medical_year2_all_generator_train_avg_3_lenexp_10_update.json', 'w') as f:
    json.dump(uk_medical_year2_training_dict, f, indent=4)