import json
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load JSON
with open(BASE_DIR / "Data/model/N_pretrain_hyperparameter_losses.json", "r") as f:
    all_losses = json.load(f)

epochs = list(range(1, 20+ 1))  # Assuming 20 epochs as per the JSON data

plt.figure(figsize=(12, 6))

colors = ['C1']  # matplotlib default colors, one per combination

# For epoch loss visualization:
plt.plot(epochs, all_losses["train_losses"], marker='o', linestyle='-', color=colors[0], label=f"train")
# Val: dashed line, cross markers, same color
plt.plot(epochs, all_losses["val_losses"], marker='x', linestyle='--', color=colors[0], label=f"val")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss For n-head, n-layer Variants")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.show()
