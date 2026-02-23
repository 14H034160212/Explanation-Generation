## The steps about data preprocessing for explanation generator
## Step 1: remove the question, options which is nan and including '<img' tag and the total rating lower than 10.
## Step 2: clean the question, options and explantion where has the html tag and extra white space.
## Step 3: remove the the explanation where there is nothing after cleaning the explanation.

import pandas as pd
import json
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tqdm import tqdm
# load_model_name = "./llama_7B_hf/llama-7b/"
# load_model_name = "./qiming_llama_7B_Cardiff_Sydney_merged_generator/"
# load_model_name = "./qiming_alpaca_7B_Cardiff_Sydney_merged_generator/"
# load_model_name = "./qiming_alpaca_7B/"
# load_model_name = "./LLaMA_7B_Cardiff_generator/"

# load_model_name = "./qiming_alpaca_7B_Cardiff_generator/"
# load_model_name_list = ["./qiming_alpaca_7B_Cardiff_generator/", "./qiming_alpaca_7B/", "./LLaMA_7B_Cardiff_generator/"]
# load_model_name_list = ["./vicuna-13b/", "./vicuna_13B_Sydney_all_generator_avg_3_lenexp_10/"]
# load_model_name_list = ["./llama_13B_hf/","./llama_2_13B_merged_all_evaluator/"]
# load_model_name_list = ["./llama_13B_hf/","./llama_2_13B_merged_all_evaluator/"]
load_model_name_list = ["./llama_2_13B_uk_medical_year1_evaluator"]
path_list = ["./PeerWiseData/Medicine/"]
# cardiff_all_question=pd.read_excel(path_list[0]+'Questions.xlsx')
# cardiff_all_question = pd.read_json(path_list[0]+"Sydney_all_generator_test_avg_3_lenexp_10.json")
# cardiff_all_question = pd.read_json(path_list[0]+"Sydney/round2/Sydney_all_generator_test_avg_3_lenexp_10_round2_random_sample_100.json")
cardiff_all_question = pd.read_json(path_list[0]+"evaluator_Test_uk_medical_year1.json")

# tag = "alpaca_7B_Cardiff_Sydney_merged_generator_"
# tag = "alpaca_7B_"
# tag1 = "alpaca_7B_Cardiff_generator_"
# tag2 = "alpaca_7B_Cardiff_test_"
# tag3 = "LLaMA_7B_Cardiff_generator_"
# tag4 = "Vicuna_13B_Cardiff_generator_"
# tag5 = "Alpaca_13B_Cardiff_generator_"

# tag4 = "Vicuna_13B_Sydney_generator_"
# tag5 = "Vicuna_13B_Sydney_generator__avg_3_lenexp_10_"
# tag4 = "Llama_2_13B_evaluator_"
# tag5 = "Llama_2_13B_uk_medical_year1_merged_evaluator_"
tag4 = "Llama_2_13B_uk_medical_year1_evaluator_"
# tag4 = "Llama_2_13B_uk_medical_year1_merged_evaluator_"

# tag6 = "gpt4_x_alpaca_13B_Cardiff_generator_"
# cardiff_all_question.rename(columns={0:'id',1:'course_id',2:'timestamp',3:'user',4:'avg_rating',5:'total_responses',6:'total_ratings',7:'top_rating_count',8:'avg_difficulty',9:'total_comments',10:'deleted',11:'answer',12:'numAlts',13:'question',14:'altA',15:'altB',16:'altC',17:'altD',18:'altE',19:'explanation'},inplace=1)
# Sydney_all_questions.rename(columns={0:'id',1:'course_id',2:'timestamp',3:'user',4:'avg_rating',5:'total_responses',6:'total_ratings',7:'top_rating_count',8:'avg_difficulty',9:'total_comments',10:'deleted',11:'answer',12:'numAlts',13:'question',14:'altA',15:'altB',16:'altC',17:'altD',18:'altE',19:'explanation'},inplace=1)
# Sydney_additionalLTISet_all_questions.rename(columns={0:'id',1:'course_id',2:'timestamp',3:'user',4:'avg_rating',5:'total_responses',6:'total_ratings',7:'top_rating_count',8:'avg_difficulty',9:'total_comments',10:'deleted',11:'answer',12:'numAlts',13:'question',14:'altA',15:'altB',16:'altC',17:'altD',18:'altE',19:'explanation'},inplace=1)

total_questions = [cardiff_all_question]
# total_list = [tag1+"cardiff_test_question_generated_explanation", 
#               tag2+"cardiff_test_question_generated_explanation", 
#               tag3+"cardiff_test_question_generated_explanation"]
total_list = [tag4+"uk_medical_year1_test"]
            #   tag5+"uk_medical_year1_test"]
            #   tag6+"cardiff_test_question_generated_explanation"
            #   ]
def load_model(model_name, eight_bit=0, device_map="auto"):
    global model, tokenizer, generator

    print("Loading "+model_name+"...")

    if device_map == "zero":
        device_map = "balanced_low_0"

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_name,
        #device_map=device_map,
        #device_map="auto",
        torch_dtype=torch.float16,
        #max_memory = {0: "14GB", 1: "14GB", 2: "14GB", 3: "14GB",4: "14GB",5: "14GB",6: "14GB",7: "14GB"},
        #load_in_8bit=eight_bit,
        #from_tf=True,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        cache_dir="cache"
    ).cuda()

    generator = model.generate

for model_id in range(len(load_model_name_list)):
    load_model(load_model_name_list[model_id])

    for i in tqdm(range(len(total_questions))):
        data = {'instruction':[],'input':[],'output':[], 'predicted_rating_score':[]}
        json_data = []
        for index, row in total_questions[i].iterrows():
            response = ""
            while response == "":
                fulltext = "Instruction: " + row["instruction"].replace("</s>", "") + " Please generate a quality rating score between 1 and 5. input " + row["input"].replace("</s>", "") + " output: "            
                
                generated_text = ""
                gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
                in_tokens = len(gen_in)
                with torch.no_grad():
                    generated_ids = generator(
                        gen_in,
                        max_new_tokens=1024,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                        num_return_sequences=1,
                        do_sample=True,
                        repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                        temperature=0.5, # default: 1.0
                        top_k = 50, # default: 50
                        top_p = 1.0, # default: 1.0
                        early_stopping=True,
                    )
                    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?

                    text_without_prompt = generated_text[len(fulltext):]

                response = text_without_prompt

                # response = response.split(human_invitation)[0]

                response = response.strip()
            data["instruction"].append(row["instruction"].replace("</s>", ""))
            data["input"].append(row["input"].replace("</s>", ""))
            explanation = ""
            if "output" in total_questions[i].columns:
                groundtruth = row["output"]
            elif "Explanation" in total_questions[i].columns:
                groundtruth = row["Explanation"]
            data["output"].append(groundtruth)
            data["predicted_rating_score"].append(response)
            
            json_data.append({"instruction":row["instruction"].replace("</s>", ""),
                            "input":row["input"].replace("</s>", ""),
                            "output":groundtruth,
                            "predicted_rating_score":response})


        with open(path_list[0]+total_list[model_id]+".json", "w") as f:
            json.dump(json_data, f, indent=4)
        
        df = pd.DataFrame(data)
        df.to_excel(path_list[0]+total_list[model_id]+".xlsx")
        print(path_list[0]+total_list[model_id]+".xlsx has been saved.")