
hp_configs = { 
            "run_name": "test_lstm-01",
            "lr": 1e-3,
            "batch_size": 128,
            "epochs": 3,
            "optimizer": 'adam', # 'adamW'
            
            "dropout": 0.3,
            
            #lstm model
            "hidden_size": 256,
            "num_layers": 2,
            "bidirectional": False,
            
            }

device = 'cpu'