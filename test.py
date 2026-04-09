import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
TRAINED_WEIGHTS_PATH = BASE_DIR / "Data/model/finetuned_model_epoch19.pt"  # Path to the finetuned model weights

def get_pretrained_max_len(pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
    # Extract the size from the position embedding weight
    return checkpoint['position_emb.weight'].shape[0]

# Model (similar to finetune)
class NbAgClassifier(nn.Module):
    def __init__(self, d_model, max_seq_len,
                 nhead=12, n_layers=6, dim_feedforward=2048, dropout=0.1,
                 pad_id=0):

        super().__init__()
        self.pad_id = pad_id

        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        self.token_emb = nn.Embedding(24, d_model, padding_idx=pad_id)
        self.physioco_emb = nn.Linear(9, d_model)
        self.position_emb = nn.Embedding(max_seq_len, d_model)
        self.segment_emb = nn.Embedding(2, d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )

        # CLS head
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, token_ids, physioco_ids, segment_ids, padding_mask):
        N, L = token_ids.shape

        token_emb = self.token_emb(token_ids)
        physioco_emb = self.physioco_emb(physioco_ids)

        position_ids = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(N, L)
        position_ids = position_ids * (~padding_mask)
        position_emb = self.position_emb(position_ids)

        segment_emb = self.segment_emb(segment_ids)

        x = token_emb + physioco_emb + position_emb + segment_emb
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # CLS token = first token
        cls_repr = x[:, 0, :]                # (N, d_model)
        logits = self.classifier(cls_repr)  # (N, 1)

        return logits, x   # x contains hidden states


# Load weights
def load_finetuned_weights(model, path):
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Finetuned weights loaded")


# Test loop
def test(model, dataloader):
    model.eval()

    all_probs = []
    all_labels = []
    all_hidden = []

    with torch.no_grad():
        for batch in dataloader:
            token_ids = batch['token_ids'].to(device)
            physioco_ids = batch['physioco_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            labels = batch['labels'].to(device)
            padding_masks = batch['padding_masks'].to(device)

            logits, hidden_states = model(
                token_ids, physioco_ids, segment_ids, padding_masks
            )

            # Probabilities (sigmoid)
            probs = torch.sigmoid(logits).squeeze(-1)   # (N,)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Hidden representations
            # shape: (N, L, d_model)
            PAD_ID = 0
            CLS_ID = 1
            SEP_ID = 2

            for i in range(token_ids.size(0)):

                tokens = token_ids[i]            # [L]
                segments = segment_ids[i]        # [L]
                padding = padding_masks[i]       # [L]
                hidden = hidden_states[i]        # [L, 768]

                # remove padding
                valid_mask = ~padding

                tokens = tokens[valid_mask]
                segments = segments[valid_mask]
                hidden = hidden[valid_mask]

                # Create filters to remove special tokens
                special_mask = (
                    (tokens != CLS_ID) &
                    (tokens != SEP_ID)
                )

                tokens = tokens[special_mask]
                segments = segments[special_mask]
                hidden = hidden[special_mask]

                # split Nb vs Ag
                nb_emb = hidden[segments == 0]
                ag_emb = hidden[segments == 1]
                
                # Store hidden representations along with label
                all_hidden.append({
                    "res_ids": tokens.cpu().numpy(),
                    "nb": nb_emb.cpu().numpy(),
                    "ag": ag_emb.cpu().numpy(),
                    "label": labels[i].item()
                })

    return np.array(all_probs), np.array(all_labels), all_hidden


if __name__ == "__main__":
    from dataset import NbAgDataset, load_data, excel_path
    from data_loader import NbAgDataLoader

    torch.manual_seed(42)

    # Load data
    train_df, finetune_df, test_df = load_data(excel_path)
    test_dataset = NbAgDataset(test_df, mlm=False)

    dataloader = NbAgDataLoader(test_dataset, batch_size=8, test=True)
    test_loader = dataloader.test_loader   # or create dedicated test loader

    # Model
    max_seq_len = get_pretrained_max_len(TRAINED_WEIGHTS_PATH)

    model = NbAgClassifier(
        d_model=768,
        max_seq_len=max_seq_len
    ).to(device)

    # Load finetuned weights
    load_finetuned_weights(model, TRAINED_WEIGHTS_PATH)

    # Run test
    probs, labels, hidden = test(model, test_loader)

    # Save outputs
    np.save("test_probs.npy", probs)
    np.save("test_labels.npy", labels)
    import pickle
    import gzip
    with gzip.open("test_hidden.pkl.gz", "wb") as f:
        pickle.dump(hidden, f)

    print("Saved probabilities, labels, and hidden representations")