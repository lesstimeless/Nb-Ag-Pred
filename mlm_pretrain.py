import torch.nn as nn
import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class NbAgMLM(nn.Module):
    def __init__(self, d_model, max_seq_len, 
                 nhead=8, n_layers=12, vocab_size=24, dim_feedforward=2048, dropout=0.1,
                 pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.physioco_emb = nn.Linear(9, d_model)  # 9 physicochemical properties
        self.position_emb = nn.Embedding(max_seq_len+10, d_model)  # +10 buffer for special tokens and potential padding
        self.segment_emb = nn.Embedding(2, d_model)  # 2 segments: Nb and Ag

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                       dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=n_layers
        )

        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, physioco_ids, segment_ids, padding_mask):
        N, L = token_ids.shape

        token_emb = self.token_emb(token_ids)
        physioco_emb = self.physioco_emb(physioco_ids)

        position_ids = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(N, L)
        position_ids = position_ids*(~padding_mask)  # Set position IDs to 0 for padding tokens
        position_emb = self.position_emb(position_ids)

        segment_emb = self.segment_emb(segment_ids)

        x = token_emb + physioco_emb + position_emb + segment_emb
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        mlm_logits = self.mlm_head(x)
        return mlm_logits


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for batch in dataloader:
        token_ids = batch['token_ids'].to(device)
        physioco_ids = batch['physioco_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        mlm_labels = batch['mlm_labels'].to(device)
        padding_masks = batch['padding_masks'].to(device)

        optimizer.zero_grad()
        mlm_logits = model(token_ids, physioco_ids, segment_ids, padding_masks)
        N, L, V = mlm_logits.shape
        loss = loss_fn(mlm_logits.view(N*L, V), mlm_labels.view(N*L))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def val_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in dataloader:
            token_ids = batch['token_ids'].to(device)
            physioco_ids = batch['physioco_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            mlm_labels = batch['mlm_labels'].to(device)
            padding_masks = batch['padding_masks'].to(device)

            mlm_logits = model(token_ids, physioco_ids, segment_ids, padding_masks)
            N, L, V = mlm_logits.shape
            loss = loss_fn(mlm_logits.view(N*L, V), mlm_labels.view(N*L))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def pretrain_mlm(train_dataset, d_model=768, nhead=8, n_layers=12,
                 max_seq_len=None, batch_size=8, lr=1e-4, epochs=10, device=None,
                 save_model_path=None):
    from data_loader import NbAgDataLoader

    # Dataloaders
    dataloader = NbAgDataLoader(train_dataset, batch_size=batch_size)
    train_dataloader = dataloader.train_loader
    val_dataloader = dataloader.val_loader

    if max_seq_len is None:
        max_seq_len = max(train_dataset[i]['seq_len'] for i in range(len(train_dataset)))

    # Model
    model = NbAgMLM(d_model=d_model, max_seq_len=max_seq_len, nhead=nhead, n_layers=n_layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Track losses
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        avg_loss = train_epoch(model, train_dataloader, optimizer, device)
        val_loss = val_epoch(model, val_dataloader, device)

        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save model
    if save_model_path is not None:
        with open("pretrain_epoch_losses.json", "w") as f:
            json.dump({"train_losses": train_losses, "val_losses": val_losses}, f, indent=2)
        torch.save(model.state_dict(), save_model_path)

    return train_losses, val_losses

if __name__ == "__main__":
    from dataset import NbAgDataset, load_data, excel_path
    torch.manual_seed(42)

    train_df, finetune_df, test_df = load_data(excel_path)

    # For pretraining, we only use the training set with mlm=True to get mlm_labels
    train_dataset = NbAgDataset(train_df, mlm=True)
    
    # First, tune hyperparameters (nhead and n_layers), before increasing epochs and saving the best model weights for finetuning
    # reduce the choices to only the best performing hyperparameters for epoch analysis (after hyperparameter tuning)
    nhead_choices = [12]
    n_layer_choices = [6]

    best_val_loss = float("inf")
    all_losses = {}

    # for hyperparameter tuning, we can use fewer epochs to save time
    for nhead in nhead_choices:
        for n_layers in n_layer_choices:
            print(f"Training variant: nhead={nhead}, n_layers={n_layers}")
            # use hyperparam_fintuned = f"best_pretrain_nhead{nhead}_nlayers{n_layers}.pt" to save model weights for best variant
            hyperparam_finetuned = None # replace this with model path after hyperparameter finetuning for epoch analysis, then use this path to load the best model weights under best epoch for finetuning later
            train_losses, val_losses = pretrain_mlm(
                train_dataset=train_dataset, 
                d_model=768, 
                nhead=nhead, 
                n_layers=n_layers,
                max_seq_len=None, 
                batch_size=8, 
                lr=1e-4, 
                epochs=5,  # Use fewer epochs for hyperparameter tuning; increase this for final training with best hyperparameters; use best epoch to save model weights for finetuning later
                device=device,
                save_model_path=hyperparam_finetuned
                )

            all_losses[f"nhead{nhead}_nlayers{n_layers}"] = {
                "train_losses": train_losses,
                "val_losses": val_losses
            }

            

# save losses to file for plotting later
with open("pretrain_hyperparameter_losses.json", "w") as f:
    json.dump(all_losses, f, indent=2)
