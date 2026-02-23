import os, json, itertools, bisect, gc

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
from accelerate import Accelerator
import accelerate
import time
import chat_generator
import chat_verifier_way2
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score

model = None
tokenizer = None
generator = None


history = []
question_msg = None
numOption_msg = None
OptionA_msg = None
OptionB_msg = None
OptionC_msg = None
OptionD_msg = None
OptionE_msg = None 
answer_msg = None
response = None

def numOptionJudgeCondition(s):
    try:
        res = int(s)
        if res >=1 and res <=5:
            return True
        else:
            return False
    except ValueError:
        return False
    
def go(global_step, msg, response, global_score_tag):
    history = []
    response2, history = chat_generator.explanationGenerator(history,global_step,msg,global_score_tag,response)
    global_score_tag = chat_verifier_way2.explanationVerifier(msg, response2)
    
    global_step += 1
    
    return response2, history, global_score_tag, global_step

if __name__ == "__main__":
    global_step = 0
    # global_score_tag = 0
    # threshold = 3
    sydney_explanation = pd.read_json("/data/qbao775/Explanation-Generation/Paul_new_data/Sydney/llama_2_13B_Sydney_merged_generator_Sydney_all_test_question_generated_explanation_bleu_bert_score_no_empty_explanation_no_s.json")
    cardiff_explanation = pd.read_json("/data/qbao775/Explanation-Generation/Paul_new_data/Cardiff/llama_2_13B_Cardiff_merged_generator_cardiff_test_question_generated_explanation_bleu_bert_score_no_empty_explanation_no_s.json")
    auckland_law_explanation = pd.read_json("/data/qbao775/Explanation-Generation/PeerWiseData/Law/llama_2_13B_Auckland_law_merged_generator_Auckland_law_test_question_generated_explanation_bleu_bert_score_no_empty_explanation_no_s.json")
    uk_medical_year1_explanation = pd.read_json("/data/qbao775/Explanation-Generation/PeerWiseData/Medicine/llama_2_13B_UK_medical_year1_merged_generator_UK_medical_year1_test_question_generated_explanation_bleu_bert_score_no_empty_explanation_no_s.json")
    uk_medical_year2_explanation = pd.read_json("/data/qbao775/Explanation-Generation/PeerWiseData/Medicine/llama_2_13B_UK_medical_year2_merged_generator_UK_medical_year2_test_question_generated_explanation_bleu_bert_score_no_empty_explanation_no_s.json")
    
    dataframe_list = [sydney_explanation, cardiff_explanation, auckland_law_explanation, uk_medical_year1_explanation, uk_medical_year2_explanation]
    flag = 5+1
    model_name = "Fine_tuned_llama_2_13B_merged_generator_flag_"+str(flag)+"_"
    name_list = ["Sydney","Cardiff","Auckland_law","UK_medical_year1", "UK_medical_year2"]
    index_flag = 0
    for data in dataframe_list:
        results_list = []
        for index, row in data.iterrows():
            msg = row["input"]
            ground_truth = row["Explanation"]
            response = row["Generated_Explanation"]
            global_score_tag = chat_verifier_way2.explanationVerifier(msg, response)
            results = {"input": msg}
            results['Explanation'] = ground_truth
            results['quality_rating_score'] = global_score_tag
            results['bleu_score'] = row['bleu_score']
            results['bert_score'] = row['bert_score']
            i = 1
            while i < flag:
                response2, history, global_score_tag2, new_global_step = go(i,msg,response,global_score_tag)
                global_step = new_global_step
                results[f'output_{i}'] = response2
                results[f'quality_rating_score_{i}'] = global_score_tag2
                results[f'bleu_score_{i}'] = sentence_bleu([ground_truth], response2)
                precision, recall, bertscore = score([response2], [ground_truth], lang="en", model_type="bert-base-uncased", verbose=False)
                bertscore = bertscore.item()
                results[f'bert_score_{i}'] = bertscore
                i = i + 1
            results_list.append(results)
        if index_flag == 0:
            output_dir = "/data/qbao775/Explanation-Generation/Paul_new_data/Sydney/"+model_name+name_list[index_flag]+'.json'
            output_dir_excel = "/data/qbao775/Explanation-Generation/Paul_new_data/Sydney/"+model_name+name_list[index_flag]+'.xlsx'
        elif index_flag == 1:
            output_dir = "/data/qbao775/Explanation-Generation/Paul_new_data/Cardiff/"+model_name+name_list[index_flag]+'.json'
            output_dir_excel = "/data/qbao775/Explanation-Generation/Paul_new_data/Cardiff/"+model_name+name_list[index_flag]+'.xlsx'
        elif index_flag == 2:
            output_dir = "/data/qbao775/Explanation-Generation/PeerWiseData/Law/"+model_name+name_list[index_flag]+'.json'
            output_dir_excel = "/data/qbao775/Explanation-Generation/PeerWiseData/Law/"+model_name+name_list[index_flag]+'.xlsx'
        elif index_flag == 3:
            output_dir = "/data/qbao775/Explanation-Generation/PeerWiseData/Medicine/"+model_name+name_list[index_flag]+'.json'
            output_dir_excel = "/data/qbao775/Explanation-Generation/PeerWiseData/Medicine/"+model_name+name_list[index_flag]+'.xlsx'
        elif index_flag == 4:
            output_dir = "/data/qbao775/Explanation-Generation/PeerWiseData/Medicine/"+model_name+name_list[index_flag]+'.json'
            output_dir_excel = "/data/qbao775/Explanation-Generation/PeerWiseData/Medicine/"+model_name+name_list[index_flag]+'.xlsx'
        
        with open(output_dir, 'w') as f:
            json.dump(results_list, f, indent=4)
            
        df = pd.DataFrame(results_list)
        df.to_excel(output_dir_excel, index=False)
        
        index_flag = index_flag + 1