import random
import numpy as np
import pandas as pd
import torch
from .datatools import utterances_to_tensors, clean_utterance_text

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
# converts raw model predictions to a series of text names
def preds_array_to_series(preds, ids_to_names):
    series = pd.Series(preds)
    def convert_to_names(plist, ids_to_names):
        plist = [ids_to_names[i] for i,item in enumerate(plist)]
        return " ".join(plist)
        
    series= series.apply(lambda x: convert_to_names(x, ids_to_names))
    return series


# func to create a .csv file for kaggle submission
def make_submission_file(model, test_csv_path, test_dataloader, INDEX_TO_CLASS,save_submission_file_path="submission.csv", device='cpu'):
    device=torch.device(device)
    model.to(device)
    model.eval()
    padding_value = test_dataloader.dataset.padding_value
    
    df = pd.read_csv(test_csv_path, index_col='ID')
    
    with torch.inference_mode():
        for inputs, targets in test_dataloader:
            preds = model(inputs.to(device)).cpu()
            
    
    preds = preds.argmax(dim=-1).long()
    targets = targets.long()
    preds, targets = list(preds), list(targets)
    masks = [target!=padding_value for target in targets]
    preds = [" ".join([INDEX_TO_CLASS[token] for token in pred[mask].tolist()]) for pred, mask in zip(preds, masks)]
    
    df["IOB Slot tags"] = preds
    df=df.drop(columns=['utterances'])
    df.to_csv(save_submission_file_path)
    
    
# def make_validation_file(model, vectorizer, val_df, ids_to_names, save_validation_file_path="validation.csv", threshold=0., device='cpu'):
#     device=torch.device(device)
#     model.to(device)
    
#     # convert sentences to tensors
#     inputs = utterances_to_tensors(val_df['UTTERANCES'], vectorizer)
    
#     model.eval()
#     with torch.inference_mode():
#         preds = model(inputs.to(device)).cpu()
    
#     preds = (preds>threshold).long().numpy()
#     val_df.to_csv("./data/val_df.csv")
#     val_df.iloc[:, 1:] = preds
#     val_df.to_csv(save_validation_file_path)
    