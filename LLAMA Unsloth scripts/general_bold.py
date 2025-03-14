import pandas as pd
import csv
import os
import shutil
import multiprocessing
from scipy.stats import pearsonr

import logging
import torch

import numpy as np

import torch.nn as nn
import pytorch_lightning as pl

from transformers import BertTokenizer

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.ao.quantization.qconfig import float_qparams_weight_only_qconfig
from transformers import AutoTokenizer, AutoModel, AutoConfig

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import torch.nn.functional as F

from datasets import Dataset, DatasetDict

from unsloth import FastLanguageModel
import torch

# %%capture
# !pip install unsloth "xformers==0.0.28.post2"

import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reproducibility
seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)

###################################################

MAX_SEQ_LEN = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = "float16" # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

def scalare(x, minimum, maximum):
    return (x - minimum) / (maximum - minimum)

# Add a lightweight regression head
class LLaMAForRegression(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.output_layer = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        # Forward pass through the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Use the last hidden state and perform mean pooling
        last_hidden_state = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size)
        pooled_output = last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)
        
        # Pass through the regression head
        regression_output = self.output_layer(pooled_output)  # Shape: (batch_size, 1)
        return regression_output
    
class RegressionSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False): 
        labels = inputs.pop("labels") # Extract labels (numeric targets) from inputs
        
        # Forward pass
        outputs = model(**inputs)
        
        # Compute regression loss (MSE)
        predictions = outputs.logits.squeeze(-1)  # Ensure shape compatibility
        loss = F.mse_loss(predictions, labels)
        
        return (loss, outputs) if return_outputs else loss

def define_trainer(model, tokenizer, train_dataset, val_dataset):
    trainer = RegressionSFTTrainer(
        model=model,  # Use the regression-enhanced model
        tokenizer=tokenizer,
        train_dataset= train_dataset, # Dataset must include "text" and "labels" fields
        eval_dataset = val_dataset, # Dataset must include "text" and "labels" fields
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=2,
        packing=False,  # False is fine for regression tasks
        args=TrainingArguments(
            num_train_epochs=5,  # NUM EPOCHS
            per_device_train_batch_size=2,
            evaluation_strategy="epoch",
            gradient_accumulation_steps=4, # Adjustable; here it's really small 60, possibly for experimenting
            warmup_steps=5,
            max_steps=None,  # Adjustable; here it's really small 60, possibly for experimenting
            learning_rate=5e-5,  #2e-4 is a bit too high, Consider starting with 5e-5 or 1e-4
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="./outputs",
            report_to="none",  # Adjust if you want to log metrics
        ),
    )
    return trainer

def define_model_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B",
        max_seq_length = MAX_SEQ_LEN,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    # Wrap the LLaMA model
    regression_model = LLaMAForRegression(model)

    #We now add LoRA adapters so we only need to update 1 to 10% of all parameters!
    model = FastLanguageModel.get_peft_model(
        regression_model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return (model, tokenizer)

def process_input(minimum, maximum, df, col):
    data = {
            "text": df['sentence'],
            "labels": scalare(df[col], minimum, maximum)
        }
    data_df = pd.DataFrame(data)

    train_data, val_data = train_test_split( data_df, test_size=0.2, random_state=42, shuffle=True )
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    return (train_dataset, val_dataset)



if __name__ == '__main__':
    testing = False
    columns = ['lang_LH_AntTemp', 'lang_LH_IFG','lang_LH_IFGorb','lang_LH_MFG','lang_LH_PostTemp','lang_LH_netw']

    inc = ''
    df = pd.read_csv(inc + 'bold_response_LH.csv')
    df_test = pd.read_csv(inc + 'train_sent.csv')

    if testing:
        print(testing)
    else: ##### CREATING PREDICTIONS
        output_file = 'corr_llama_bold_5ep.csv'
        with open(output_file, 'w', newline='') as csvfile:
            wrt = csv.writer(csvfile)
            wrt.writerow(['column', 'correlation'])
            for col in columns:
                maximum = df[col].max()
                minimum = df[col].min()

                train_dataset, val_dataset = process_input(minimum, maximum, df, col)

                #######################################
                model, tokenizer = define_model_tokenizer()
                trainer = define_trainer(model, tokenizer, train_dataset, val_dataset)
                trainer_stats = trainer.train()
                #######################################

                FastLanguageModel.for_inference(model)

                test_sentences = list(df_test['sentence'].to_records(index=False))
                test_values = list(df_test['total_fix_dur'].to_records(index=False))
                

                # Perform inference
                with torch.no_grad():  # No gradient computation needed during inference
                    outputs = model(**test_sentences)

                # Extract regression outputs (predicted values)
                predictions = outputs.logits.squeeze(-1)  # Ensure logits have the correct shape
                predictions_list = predictions.tolist()

                correlation, _ = pearsonr(predictions_list, test_values)
                wrt.writerow([col, correlation])
            

                

