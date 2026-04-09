from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
import torch

torch.manual_seed(42)

def collate_fn_pretrain(batch):
    batch_token_ids = pad_sequence([item['token_ids'] for item in batch], batch_first=True, padding_value=0)
    batch_physioco_ids = pad_sequence([item['physioco_ids'] for item in batch], batch_first=True, padding_value=0.0)
    batch_segment_ids = pad_sequence([item['segment_ids'] for item in batch], batch_first=True, padding_value=0)
    batch_mlm_labels = pad_sequence([item['mlm_labels'] for item in batch], batch_first=True, padding_value=-100)
    batch_seq_len = torch.tensor([item['seq_len'] for item in batch], dtype=torch.long)

    # Attention mask for padded tokens (True for padding positions, False for real tokens)
    batch_padding_masks = torch.arange(batch_token_ids.size(1), device=batch_token_ids.device)[None, :] >= batch_seq_len[:, None]

    return {
        'token_ids': batch_token_ids,
        'physioco_ids': batch_physioco_ids,
        'segment_ids': batch_segment_ids,
        'mlm_labels': batch_mlm_labels,
        'seq_len': batch_seq_len,
        'padding_masks': batch_padding_masks
    }

def collate_fn_finetune(batch):
    batch_token_ids = pad_sequence([item['token_ids'] for item in batch], batch_first=True, padding_value=0)
    batch_physioco_ids = pad_sequence([item['physioco_ids'] for item in batch], batch_first=True, padding_value=0.0)
    batch_segment_ids = pad_sequence([item['segment_ids'] for item in batch], batch_first=True, padding_value=0)
    batch_labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    batch_seq_len = torch.tensor([item['seq_len'] for item in batch], dtype=torch.long)

    # Attention mask for padded tokens (True for padding positions, False for real tokens)
    batch_padding_masks = torch.arange(batch_token_ids.size(1), device=batch_token_ids.device)[None, :] >= batch_seq_len[:, None]

    return {
        'token_ids': batch_token_ids,
        'physioco_ids': batch_physioco_ids,
        'segment_ids': batch_segment_ids,
        'labels': batch_labels,
        'seq_len': batch_seq_len,
        'padding_masks': batch_padding_masks
    }

class NbAgDataLoader():
    def __init__(self, dataset, batch_size=8, shuffle=True, test=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        if test:
            self.test = dataset
            self.test_loader = DataLoader(self.test, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn_finetune)
        else:
            self.train, self.val = self.train_val_split(dataset)
            self.train_loader = self.get_dataloader(train=True)
            self.val_loader = self.get_dataloader(train=False)
        
    def train_val_split(self, dataset, val_ratio=0.1):
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        return random_split(dataset, [train_size, val_size])

    def get_dataloader(self, train):
        # For pretraining, we use collate_fn_pretrain which includes mlm_labels; for finetuning/testing, we use collate_fn_finetune which includes labels
        if train:
            return DataLoader(self.train, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=collate_fn_finetune)
        else:  
            return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn_finetune)
        
# for testing
if __name__ == "__main__":
    from dataset import NbAgDataset, load_data, excel_path
    train_df, finetune_df, test_df = load_data(excel_path)
    dataset = NbAgDataset(train_df, mlm=True)
    dataloader = NbAgDataLoader(dataset)
    for batch in dataloader.train_loader:
        print(batch['token_ids'].shape, batch['physioco_ids'].shape, batch['segment_ids'].shape, batch['mlm_labels'].shape, batch['seq_len'].shape)
        break