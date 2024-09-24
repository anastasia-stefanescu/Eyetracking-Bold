import pandas as pd
import csv
import os
import shutil

import logging
import torch
from scipy.stats import pearsonr

import numpy as np

import torch.nn as nn
import pytorch_lightning as pl

from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.metrics import mean_squared_error

#from pprint import pprint

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModel, AutoConfig, TrainingArguments

import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reproducibility
seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)

MAX_SEQ_LEN = 256  # Adjust the maximum length of the sequence to your needs
BATCH_SIZE = 16    # Adjust the batch size to your requirements


##########################################

def scalare(x, minimum, maximum):
    return (x - minimum) / (maximum - minimum)

class CustomDataset(Dataset):
    def __init__(self, df):
        self.samples = [{"content": row['word'], "class": scalare(row['total_fix_dur'], minimum, maximum)} for _, row in df.iterrows()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
    
class CustomCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __call__(self, input_batch: list[dict]) -> dict:
        words = [instance['content'] for instance in input_batch]
        targets = [instance['class'] for instance in input_batch]

        # Tokenize the words with padding and truncation
        tokenized_batch = self.tokenizer(
            words,
            padding=True,               # Pad to the longest word in the batch
            max_length=self.max_seq_len, # Truncate if the word exceeds max length
            truncation=True,
            return_tensors="pt"          # Return the tokenized inputs as PyTorch tensors
        )

        # Convert the regression targets to a tensor
        targets_tensor = torch.tensor(targets, dtype=torch.float)

        return {
            "input_ids": tokenized_batch["input_ids"],
            "attention_mask": tokenized_batch["attention_mask"],
            "targets": targets_tensor
        }
    
###############################
#################################

class BERTModel(pl.LightningModule):
    def __init__(self, model_name: str, lr: float = 2e-5, sequence_max_length: int = MAX_SEQ_LEN):
        super().__init__()

        self.val_preds = []
        self.val_labels = []

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the model configuration and set output_hidden_states=True
        self.model_config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        
        self.model = AutoModel.from_pretrained(model_name, config = self.model_config)
        self.output_layer = nn.Linear(self.model.config.hidden_size, 1)  # One output unit for regression
        self.loss_fct = nn.MSELoss()
        self.lr = lr
        self.save_hyperparameters()

        if using_only_layer5 == 1:
            #Fixed weights - pt toate 
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.encoder.layer[4].parameters():
                param.requires_grad = True
            
            # Ensure the regression output layer is trainable
            for param in self.output_layer.parameters():
                param.requires_grad = True

        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU is not available, using CPU.")

    def forward(self, input_ids, attention_mask):
        # Pass input_ids and attention_mask to the base model
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        if using_only_layer5 == 1:
            # Extract the hidden states from the 5th transformer block
            hidden_states = output.hidden_states
            layer_5_output = hidden_states[4]  # 5th block's output

            # Use the [CLS] token's representation from the 5th block for regression
            cls_embedding = layer_5_output[:, 0, :]  # [CLS] token embedding
            
            # Pass through the output layer (regression head)
            return self.output_layer(cls_embedding).flatten()

        else:
            # Apply the output layer to the pooled output of the model
            return self.output_layer(output.pooler_output).flatten()


    def training_step(self, batch, batch_idx):
        
        # Prepare the tokenized batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        ground_truth = batch['targets'].float()
        
        # Forward pass
        prediction = self(input_ids=input_ids, attention_mask=attention_mask)
        
        loss = self.loss_fct(prediction, ground_truth)
        mae = torch.mean(torch.abs(prediction - ground_truth))

        self.log("train_loss", loss.detach().cpu().item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    
    def validation_step(self, batch, batch_idx):
        # Prepare the tokenized batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        ground_truth = batch['targets'].float()
        
        # Forward pass
        prediction = self(input_ids=input_ids, attention_mask=attention_mask)
        
        loss = self.loss_fct(prediction, ground_truth)
        mae = torch.mean(torch.abs(prediction - ground_truth))

        self.log("val_loss", loss.detach().cpu().item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=True)

         # Save predictions and ground truth for later correlation calculation
        self.val_preds.extend(prediction.detach().cpu().numpy())
        self.val_labels.extend(ground_truth.detach().cpu().numpy())

        return {"loss": loss}
    
    def on_validation_epoch_end(self):
        # Convert stored lists of predictions and labels to numpy arrays
        val_preds = np.array(self.val_preds)
        val_labels = np.array(self.val_labels)

        # Calculate Pearson correlation
        if len(val_preds) > 0 and len(val_labels) > 0:
            pearson_correlation, _ = pearsonr(val_preds, val_labels)
        else:
            pearson_correlation = 0.0

        # Log the Pearson correlation
        self.log("val_pearson_corr", pearson_correlation, prog_bar=True)

        # Clear the stored predictions and labels for the next epoch
        self.val_preds = []
        self.val_labels = []

    def configure_optimizers(self):
        return AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)



def predict(model, tokenizer, word):
    # Tokenize the single word input
    tokenized_input = tokenizer(
        word,
        padding=True,
        max_length=10,  # You can reduce the max length for a single word
        truncation=True,
        return_tensors="pt"
    )

    # Move tokenized input to the same device as the model
    tokenized_input = {key: value.to(next(model.parameters()).device) for key, value in tokenized_input.items()}

    with torch.no_grad():  # Turn off gradient computation for inference
        predictions = model(input_ids=tokenized_input['input_ids'], attention_mask=tokenized_input['attention_mask'])

    prediction_score = predictions.mean().item()  # Handle batch dimension if needed

    return prediction_score


def evaluate_model_on_tests(model, lista, output_file):
    model.eval()  # Set the model to evaluation mode.

    device = next(model.parameters()).device  # Get the device of the model's parameters.

    with torch.no_grad(), open(output_file, 'a', newline='') as csvfile:  # Inference mode, no gradients needed.
        lung = len(lista)

        wrt = csv.writer(csvfile)
        wrt.writerow(['index', 'sent_id', 'word', 'prediction']) #""",'actual'"""
        for i in range(lung):
            word = lista[i][1]
            sent_id = lista[i][0]
            rez = predict(model, tokenizer, word)
            scalat = rez * (maximum - minimum) + minimum
            #file.write(f"{i} {sent_id} {word} {int(scalat)} {test_df.loc[i, 'mean_fix_dur']} \n")

            wrt.writerow([i, sent_id, word, int(scalat)]) # """,test_df.loc[i, 'mean_fix_dur']""" 


def get_words():
    df = pd.read_csv('/Users/anastasiastefanescu/Documents/dataseturi eyetracking/bold_response_LH.csv')

    sentences = df['sentence'].tolist()
    n = len(sentences)
    indexed = []
    for i in range(n):
        words = sentences[i].split(" ")
        for word in words:
            indexed.append((i, word))
    return indexed


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)  # Ensures safe multiprocessing on MacOS


    ###############################

    inc = '/Users/anastasiastefanescu/Documents/dataseturi eyetracking/'
    folder = 'datasets/zuco/'
    train_df = pd.read_csv(inc + folder + 'train_dataset.csv')
    test_df = pd.read_csv(inc + folder + 'test_dataset.csv')
    val_df = pd.read_csv(inc + folder + 'val_dataset.csv')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    custom_collator = CustomCollator(tokenizer, MAX_SEQ_LEN)
    
    ##########################################
    minimum = train_df['total_fix_dur'].min()
    # minimum = min(minimum, test_df['total_fix_dur'].min())
    # minimum = min(minimum, val_df['total_fix_dur'].min())
    maximum = train_df['total_fix_dur'].max()
    # maximum = max(maximum, test_df['total_fix_dur'].max())
    # maximum = max(maximum, val_df['total_fix_dur'].max())
    
        
    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)
    test_dataset = CustomDataset(test_df)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collator, num_workers = 2, pin_memory=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collator, num_workers = 2, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collator, num_workers = 2)


    using_only_layer5 = 1
    
    MODEL_PATH = "bert-base-uncased"  # or your custom model path
    model = BERTModel(model_name=MODEL_PATH)

    trainer = pl.Trainer(
        devices=-1,  # Comment this when training on cpu
        accelerator="gpu",
        max_epochs=10,  # Set this to -1 when training fully
        limit_train_batches=10,  # Uncomment this when training fully
        limit_val_batches=5,  # Uncomment this when training fully
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        log_every_n_steps = 50
    )

    #Training
    trainer.fit(model, train_dataloader, validation_dataloader)

    pth = inc+'lightning_logs/'
    if os.path.exists(pth):
        shutil.rmtree(pth)  # Recursively delete the log directory
        print(f"Deleted log directory: {pth}")

    data_test = list(test_df.to_records(index=False))
    data_test_str = [(str(elem[1]), str(elem[2])) for elem in data_test]
    output_file = "pred_zuco_test.csv"
    evaluate_model_on_tests(model, data_test_str, output_file)

    ####afisam corelatia cu datele de test
    test_pred = pd.read_csv(inc + output_file)
    test = pd.read_csv(inc + folder + 'test_dataset.csv')

    print(f"Corelatia dintre date reale si predictii: {test_pred['prediction'].corr(test['total_fix_dur'])}")

    ############

    output_file = "pred_zuco_bold.csv"
    data_test_str = get_words()
    evaluate_model_on_tests(model, data_test_str, output_file)

   
    #####calculam suma pe propozitii a timpilor prezisi pt bold

    df = pd.read_csv(inc + 'bold_response_LH.csv')
    pred = pd.read_csv(inc + 'pred_zuco_bold.csv')

    file_for_sentences = 'pred_timpi_prop_zuco.csv'

    with open(file_for_sentences, 'a', newline='') as csvfile: 
        sentences = df['sentence'].tolist()
        n = len(sentences)
        indexed = []
        wrt = csv.writer(csvfile)
        wrt.writerow(['sent_id', 'sentence', 'total_time_prediction'])
        for i in range(n):
            words = sentences[i].split(" ")
            m = len(words)
            sum_total = 0
            for j in range(m):
                sum_total += pred.loc[i, 'prediction']
            wrt.writerow([i, sentences[i], sum_total])
    
    df = pd.read_csv(inc + 'bold_response_LH.csv')
    pred_upd = pd.read_csv(inc + file_for_sentences)

    columns = df.columns
    t = 'total_time_prediction'

    with open('corelatii_eye_bold.csv', 'a', newline='') as csvfile: 
        wrt = csv.writer(csvfile)
        wrt.writerow(['bold_column', 'corr_total_time']) #, 'corr_mean_time'
        for c in columns:
            if c != 'item_id' and c != 'sentence':
                wrt.writerow([c, pred_upd[t].corr(df[c])]) #, pred[m].corr(df[c])
                print(f"{c} -> corr: {pred_upd[t].corr(df[c])}")




    
    
