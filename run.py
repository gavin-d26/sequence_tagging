import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import scripts.configs as configs
import scripts.datatools as datatools
import scripts.models as models
import scripts.train as train
import scripts.predict as predict

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.use_deterministic_algorithms(True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_csv_path')
    parser.add_argument('test_csv_path')
    parser.add_argument('submission_csv_path')
    
    args = parser.parse_args()
    
    # parse script args
    train_csv_path=args.train_csv_path
    test_csv_path=args.test_csv_path
    submission_csv_path=args.submission_csv_path
    
    # set environment variables
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # get train and val dataframes
    train_df, val_df, CLASS_TO_INDEX = datatools.preprocess_raw_training_file(train_csv_path)
    test_df = pd.read_csv(test_csv_path, index_col='ID')
    INDEX_TO_CLASS = {v:k for k,v in CLASS_TO_INDEX.items()}
    # create vectorizer and dataloaders
    train_loader, val_loader, test_loader = datatools.create_dataloaders(train_df, val_df, test_df, CLASS_TO_INDEX, configs.hp_configs['batch_size'])
    
    # initialize model
    # model = models.RelationClassifierPro(train_loader.dataset[0][0].shape[1], 
    #                                      out_features=len(CLASS_TO_INDEX),
    #                                      dropout=configs.hp_configs['dropout'])
    
    model = models.SequenceTagger(input_size=train_loader.dataset[0][0].shape[1], 
                                   hidden_size=configs.hp_configs['hidden_size'], 
                                   output_size=len(CLASS_TO_INDEX),
                                   num_layers=configs.hp_configs['num_layers'], 
                                   bidirectional=configs.hp_configs['bidirectional'], 
                                   dropout=configs.hp_configs['dropout'])
    
    train.train_func(
        model,
        train_loader,
        val_loader,
        hp_config=configs.hp_configs,
        device=configs.device,
        wandb_flag=False
    )
    
    predict.make_submission_file(model,
                                 test_csv_path, 
                                 test_loader,
                                 INDEX_TO_CLASS, 
                                 save_submission_file_path=submission_csv_path, 
                                 device=configs.device)
    
    
if __name__=="__main__":
    main()
    
    