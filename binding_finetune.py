import torch
import torch.nn as nn
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
PRETRAINED_MODEL_PATH = BASE_DIR / "Data/model/X_epoch14_pretrain_nhead12_nlayers6.pt"  # Path to the pretrained MLM model weights

def get_pretrained_max_len(pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
    # Extract the size from the position embedding weight
    return checkpoint['position_emb.weight'].shape[0]

def load_pretrained_weights(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path, map_location=device, weights_only=True)

    model_dict = model.state_dict()

    # remove MLM head weights
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print("Pretrained weights loaded (excluding MLM head)")

class NbAgBindClassifier(nn.Module):
    def __init__(self, d_model, max_seq_len,
                 nhead=12, n_layers=6, vocab_size=24,
                 dim_feedforward=2048, dropout=0.1, pad_id=0):

        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.physioco_emb = nn.Linear(9, d_model)
        self.position_emb = nn.Embedding(max_seq_len, d_model)
        self.segment_emb = nn.Embedding(2, d_model)

        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)

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

        # Classification head (CLS)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, token_ids, physioco_ids, segment_ids, padding_mask):
        N, L = token_ids.shape

        token_emb = self.token_emb(token_ids)
        physioco_emb = self.physioco_emb(physioco_ids)

        position_ids = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(N, L)
        position_emb = self.position_emb(position_ids)

        segment_emb = self.segment_emb(segment_ids)

        x = token_emb + physioco_emb + position_emb + segment_emb
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # CLS token (index 0)
        cls_output = x[:, 0, :]

        logits = self.classifier(cls_output).squeeze(-1)

        return logits
    

# Train, validation across epochs 
def train_epoch(model, dataloader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0

    for batch in dataloader:
        token_ids = batch['token_ids'].to(device)
        physioco_ids = batch['physioco_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        labels = batch['labels'].float().to(device)
        padding_masks = batch['padding_masks'].to(device)

        optimizer.zero_grad()

        logits = model(token_ids, physioco_ids, segment_ids, padding_masks)

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def val_epoch(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            token_ids = batch['token_ids'].to(device)
            physioco_ids = batch['physioco_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            labels = batch['labels'].float().to(device)
            padding_masks = batch['padding_masks'].to(device)

            logits = model(token_ids, physioco_ids, segment_ids, padding_masks)

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def compute_global_pos_weight(dataset, device):
    labels = torch.tensor([sample['label'] for sample in dataset])
    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()
    
    if num_pos == 0:
        return torch.tensor(1.0, device=device)
    
    # Correct relative ratio for BCEWithLogitsLoss
    return torch.tensor(1160/111000, device=device)

def finetune_binding(finetune_dataset, pretrained_path,
                     d_model=768, batch_size=8, lr=1e-5, epochs=20, device=None):

    from data_loader import NbAgDataLoader

    dataloader = NbAgDataLoader(finetune_dataset, batch_size=batch_size)

    train_loader = dataloader.train_loader
    val_loader = dataloader.val_loader

    pretrained_max_len = get_pretrained_max_len(pretrained_path)

    model = NbAgBindClassifier(d_model=d_model, max_seq_len=pretrained_max_len).to(device)

    # load pretrained MLM weights
    load_pretrained_weights(model, pretrained_path)

    # Freeze all layers except the classification head 
    for param in model.transformer.parameters(): 
        param.requires_grad = False 
    for param in model.token_emb.parameters(): 
        param.requires_grad = False 
    for param in model.position_emb.parameters(): 
        param.requires_grad = False 
    for param in model.segment_emb.parameters(): 
        param.requires_grad = False 
    for param in model.physioco_emb.parameters(): 
        param.requires_grad = False 

    # Unfreeze last 2 transformer layer 
    for param in model.transformer.layers[-1].parameters(): 
        param.requires_grad = True 
    for param in model.transformer.layers[-2].parameters(): 
        param.requires_grad = True 

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr)

    best_val_loss = float("inf")
    best_epoch = 0
    train_losses = []
    val_losses = []

    pos_weight = compute_global_pos_weight(finetune_dataset, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = val_epoch(model, val_loader, device, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"finetuned_model_epoch{best_epoch}.pt")

    print(f"Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    # For plotting loss curves
    with open("finetuned_epoch_losses.json", "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f, indent=2)

if __name__ == "__main__":
    import torch
    from dataset import NbAgDataset, load_data, excel_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dfs, finetune_dfs, test_df = load_data(excel_path)

    finetune_dataset = NbAgDataset(finetune_dfs, mlm=False)

    finetune_binding(
        finetune_dataset=finetune_dataset,
        pretrained_path=PRETRAINED_MODEL_PATH,
        device=device,
        epochs=20 # set to 20 to find best epoch and weights for test
    )