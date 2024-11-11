import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from .datatools import RelationExtractionDataset, create_dataloaders
import wandb
from seqeval.metrics import f1_score, classification_report
from seqeval.scheme import IOB2

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# class to compute  F1 Score
class F1Score():
    def __init__(self, padding_value, INDEX_TO_CLASS):
        self.padding_value=padding_value
        self.preds=[]
        self.targets=[]
        self.INDEX_TO_CLASS=INDEX_TO_CLASS
        for index in INDEX_TO_CLASS.keys():
            if len(INDEX_TO_CLASS[index])>1:
                INDEX_TO_CLASS[index]=INDEX_TO_CLASS[index][0] + "-" + INDEX_TO_CLASS[index][2:]
    
    
    # append new batches    
    def update(self, preds, targets):
        preds = preds.argmax(dim=-1).long()
        targets = targets.long()
        preds, targets = list(preds), list(targets)
        masks = [target!=self.padding_value for target in targets]
        preds = [[self.INDEX_TO_CLASS[token] for token in pred[mask].tolist()] for pred, mask in zip(preds, masks)]
        targets = [[self.INDEX_TO_CLASS[token] for token in target[mask].tolist()] for target, mask in zip(targets, masks)]
        self.preds.extend(preds)
        self.targets.extend(targets)
    
    
    # used to compute F1 Score at the end of an epoch
    def compute(self):
        result = f1_score(self.targets, self.preds, mode='strict', scheme=IOB2, )
        self.preds=self.targets=None
        return result

# function to compute sequence loss
def sequence_loss_fn(preds, targets, padding_value):
    preds = preds.view(-1, preds.shape[-1])
    targets = targets.view(-1)
    preds, targets = preds[targets!=padding_value], targets[targets!=padding_value]
    loss = F.cross_entropy(preds, targets.long(), reduction='mean')
    return loss


# function to train the pytorch model        
def train_func(
    model,
    train_loader,
    val_loader,
    hp_config,
    device='cpu',
    wandb_flag=False
    ):
    device = torch.device(device)
    model.to(device=device)

    run_name = hp_config['run_name']
    epochs = hp_config['epochs']
    lr=hp_config['lr']
    optimizer=hp_config['optimizer']
    
    
    if optimizer=='adam':
        optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer=='adamw':
        optimizer=torch.optim.AdamW(model.parameters(), lr=lr)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader))
    
    # padding value
    padding_value=train_loader.dataset.padding_value
    
    train_acc = F1Score(padding_value=padding_value, INDEX_TO_CLASS=train_loader.dataset.INDEX_TO_CLASS)
    val_acc = F1Score(padding_value=padding_value, INDEX_TO_CLASS=val_loader.dataset.INDEX_TO_CLASS)
    
    max_val_acc=0
    max_epoch=0
    
    
    if wandb_flag is True:
        wandb.init(project="sequence_tagging", name=run_name, config=hp_config)
        
    print(f"Starting Training: {run_name}")
    
    for epoch in tqdm(range(epochs)):
        print(f"-------- Epoch {epoch} --------")
        
        train_loss=[]
        val_loss=[]
        
        # train on train set
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            preds = model(inputs)
            loss = sequence_loss_fn(preds, targets, padding_value)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_acc.update(preds.detach().cpu(), targets)
            train_loss.append(loss.detach().cpu())
            scheduler.step()
        
        
        # evaluate on val set
        model.eval()
        with torch.inference_mode():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                preds = model(inputs)
                loss=sequence_loss_fn(preds, targets, padding_value)
                val_loss.append(loss.cpu())
                val_acc.update(preds.detach().cpu(), targets)
                
                
        metrics = {
            "train_loss":sum(train_loss)/len(train_loss),
            "train_acc":train_acc.compute(),
            "val_loss":sum(val_loss)/len(val_loss),
            "val_acc":val_acc.compute(),
        }
        
        if wandb_flag is True:
            wandb.log({
                'Loss/train': metrics['train_loss'],
                'Loss/val': metrics['val_loss'],
                'Accuracy/train': metrics['train_acc'],
                'Accuracy/val': metrics['val_acc'],
                'epoch': epoch
            })
        
        if metrics["val_acc"]>max_val_acc:
            if not os.path.isdir("./checkpoints"):
                os.mkdir("./checkpoints")
            torch.save(model.state_dict(), "./checkpoints/model.pt")
            max_val_acc=metrics["val_acc"]
            max_epoch=epoch
        
        print(f'train_loss: {metrics["train_loss"]:.2f}   val_loss: {metrics["val_loss"]:.2f}   train_acc: {metrics["train_acc"]:.2f} \
                val_acc": {metrics["val_acc"]:.2f}')
    
    print(f"best model at epoch: {max_epoch}")
    
        
    # model.to(device=torch.device('cpu'))
    # model.load_state_dict(torch.load("./checkpoints/model.pt", map_location="cpu"))
        
        
        
        
        
        
        
        
        
            
            
            
        
            
            
            
    

