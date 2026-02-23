import pandas as pd

# 读取json文件
data = pd.read_json("./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json")

# 对于数据进行随机抽样
sample_data = data.sample(n=5000, random_state=1) # random_state确保了你每次得到相同的随机样本

# 将抽样数据保存为json文件
sample_data.to_json("./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10_sample_5000.json", orient='records')