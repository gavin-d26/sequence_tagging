
hp_configs = { 
            "run_name": "test_lstm-08",
            "lr": 1e-3,
            "batch_size": 128,
            "epochs": 100,
            "optimizer": 'adam', # 'adamW'
            
            "dropout": 0.4,
            
            #lstm model
            "hidden_size": 128,
            "num_layers": 1,
            "bidirectional": True,
            "notes": "using glove embeddings inaddition to spacy embeddings - 50->100d, dropout before fc layer",
            
            }

device = 'cpu'