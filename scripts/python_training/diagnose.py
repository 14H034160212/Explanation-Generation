import sys
import os

print("--- Diagnostics ---")
print(f"Python: {sys.executable}")
print(f"CWD: {os.getcwd()}")

try:
    import torch
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
except Exception as e:
    print(f"Torch Import Error: {e}")

try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}")
except Exception as e:
    print(f"Transformers Import Error: {e}")

try:
    import peft
    print(f"Peft Version: {peft.__version__}")
except Exception as e:
    print(f"Peft Import Error: {e}")

try:
    import bert_score
    print("bert_score imported")
except Exception as e:
    print(f"bert_score Import Error: {e}")

try:
    import nltk
    print("nltk imported")
except Exception as e:
    print(f"nltk Import Error: {e}")

data_path = "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/UK_medical_year1_all_generator_train_avg_3_lenexp_10_update.json"
if os.path.exists(data_path):
    print(f"Data Path Exists: {data_path}")
else:
    print(f"!! DATA PATH MISSING !!: {data_path}")

base_model = "/data/qbao775/Explanation-Generation/vicuna-13b"
if os.path.exists(base_model):
    print(f"Base Model Exists: {base_model}")
else:
    print(f"!! BASE MODEL MISSING !!: {base_model}")

print("--- End Diagnostics ---")
