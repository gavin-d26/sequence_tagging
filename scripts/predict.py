import random
import numpy as np
import pandas as pd
import torch
from .datatools import utterances_to_tensors, clean_utterance_text

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


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
    