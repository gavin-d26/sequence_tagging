import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from spacy.cli import download
from spacy.util import is_package
import torchtext

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# load spacy model
def load_spacy_model(model_name="en_core_web_md"):
    # Check if the model is already installed
    if not is_package(model_name):
        print(f"Model '{model_name}' not found. Downloading...")
        download(model_name)

    # Load the model
    return spacy.load(model_name)

nlp=load_spacy_model(model_name="en_core_web_md")

# func to clean text sentences
def clean_utterance_text(series):
    series=series.str.strip()
    series=series.str.lower()
    series=series.apply(lambda x: ''.join((item for item in x if not item.isdigit())))
    # series=pd.Series(list(nlp.pipe(series.tolist())))
    # series=series.apply(lambda doc: [word.lemma_ for word in doc])
    # series=series.apply(lambda doc: " ".join(doc))
    return series


# converts .csv file to train and val Dataframes. Also indexes the classes
def preprocess_raw_training_file(hw_csv_file):
    df = pd.read_csv(hw_csv_file)
    df = df.drop(columns=['ID'])
    
    # df['IOB Slot tags'] = df['IOB Slot tags'].str.split(" ")
    df['IOB Slot tags'] = df['IOB Slot tags'].apply(lambda x: x.replace('-', '_'))
    
    CLASS_TO_INDEX = {c: i for i, c in enumerate(df['IOB Slot tags'].str.split(" ").explode().unique())}
    
    train_df, val_df = train_test_split(df, test_size=0.20, random_state=0)
    return train_df, val_df, CLASS_TO_INDEX

    
# used to convert raw utterances to tensors for input to model (can be (B, S, embed_dim)!!)
def utterances_to_tensors(utterance_series):
    glove = torchtext.vocab.GloVe(name = "6B", dim=100)
    # Tokenize sentences and get embeddings
    def get_embeddings(sentence):
        doc = nlp(sentence)
        return [np.concatenate((token.vector, glove[token.text].numpy())) for token in doc]
    
    # apply the function to the series
    embeddings = utterance_series.apply(get_embeddings)
    
    # convert embeddings to tensors
    embedding_tensors = [torch.tensor(embed) for embed in embeddings]
    
    # pad the sequences
    padded_embeddings = pad_sequence(embedding_tensors, batch_first=True, padding_value=0)
    return padded_embeddings


# used to convert IOB Slot tags to tensors for target to model
def tags_to_tensors(tag_series, CLASS_TO_INDEX):
    def get_tag_indices(tags):
        return [CLASS_TO_INDEX[tag] for tag in tags.split(" ")]
    
    tag_indices = tag_series.apply(get_tag_indices)
    tag_tensors = [torch.tensor(tags) for tags in tag_indices]
    padded_tags = pad_sequence(tag_tensors, batch_first=True, padding_value=len(CLASS_TO_INDEX))
    return padded_tags.float()


# used to create dummy targets for the test dataset. used to indicate padding tokens
def make_dummy_targets(utterance_series):
    def get_dummy_targets(sentence):
        return [0 for token in sentence.split(" ")]
    
    dummy_targets = utterance_series.apply(get_dummy_targets)
    dummy_targets = [torch.tensor(targets) for targets in dummy_targets]
    padded_targets = pad_sequence(dummy_targets, batch_first=True, padding_value=1)
    return padded_targets.float()
    

# torch subclass for the dataset
class RelationExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, df, CLASS_TO_INDEX, dataset_type):
        super().__init__()
        self.CLASS_TO_INDEX = CLASS_TO_INDEX
        self.INDEX_TO_CLASS = {k:i for i,k in CLASS_TO_INDEX.items()}
        
        # preprocessing
        df['utterances'] = clean_utterance_text(df['utterances'])
        
        # convert utternances to tensors
        self.inputs = torch.FloatTensor(utterances_to_tensors(df['utterances']))
        
        if dataset_type in ['train', 'validation']:
            self.targets = torch.FloatTensor(tags_to_tensors(df['IOB Slot tags'], CLASS_TO_INDEX))
            self.padding_value = len(CLASS_TO_INDEX)
        
        if dataset_type=='test':
            self.targets = make_dummy_targets(df['utterances'])
            self.padding_value = 1
    
    
    def __getitem__(self, index):
        return self.inputs[index, :], self.targets[index, :]
    
    
    def __len__(self):
        return len(self.inputs)
        

# seed the workers
def worker_init_fn(worker_id):
    seed=0
    np.random.seed(seed+worker_id)
    random.seed(seed+worker_id)


# creates dataloaders for train and val datasets, Note: it no longer uses vectorizer
def create_dataloaders(train_df, val_df, test_df, CLASS_TO_INDEX, batch_size=32):
    #create datasets
    train_dataset = RelationExtractionDataset(train_df, CLASS_TO_INDEX, dataset_type='train')
    val_dataset = RelationExtractionDataset(val_df, CLASS_TO_INDEX, dataset_type='validation')
    test_dataset = RelationExtractionDataset(test_df, CLASS_TO_INDEX, dataset_type='test')
    
    #create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False, worker_init_fn=worker_init_fn)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=2, pin_memory=False, worker_init_fn=worker_init_fn)
    return train_loader, val_loader, test_dataset


        
        
        
        