import numpy as np
from pathlib import Path
import pickle
import gzip
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

BASEDIR = Path(__file__).resolve().parent
REPRESENTATION_PATH = BASEDIR / "Data/Processed/"

# Load hidden representations, labels and probs
def load_test_outputs(path, expanded = False, final =False):
    if final:
        probs = np.load(path / "test_probs.npy")
        labels = np.load(path / "test_labels.npy")
        with gzip.open(path / "test_hidden.pkl.gz", "rb") as f:
            hidden_data = pickle.load(f)
    elif expanded:
        probs = np.load(path / "X_test_probs.npy")
        labels = np.load(path / "X_test_labels.npy")
        with gzip.open(path / "X_test_hidden.pkl.gz", "rb") as f:
            hidden_data = pickle.load(f)
    else:
        probs = np.load(path / "N_test_probs.npy")
        labels = np.load(path / "N_test_labels.npy")
        with gzip.open(path / "N_test_hidden.pkl.gz", "rb") as f:
            hidden_data = pickle.load(f)
    return probs, labels, hidden_data

def find_threshold_with_constraint(probs, labels, max_fpr=0.2):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    
    valid = fpr <= max_fpr
    idx = np.argmax(tpr[valid])
    
    return thresholds[valid][idx]

def calc_metrics(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    TP = np.sum((preds == 1) & (labels == 1))
    TN = np.sum((preds == 0) & (labels == 0))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))
    
    accuracy = (TP + TN) / len(labels)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    return accuracy, precision, recall, f1_score, specificity

if __name__ == "__main__":
    N_probs, N_labels, N_hidden_data = load_test_outputs(REPRESENTATION_PATH)
    X_probs, X_labels, X_hidden_data = load_test_outputs(REPRESENTATION_PATH, expanded=True)
    probs, labels, hidden_data = load_test_outputs(REPRESENTATION_PATH, final=True)
    # print(N_probs.shape, N_labels.shape, len(N_hidden_data))
    # print(N_hidden_data[0].keys())  # Should show 'res_ids', 'nb', 'ag', 'label'
    # print(N_labels)
    # print(N_probs)

    # Find optimal thresholds
    N_optimal_threshold = find_threshold_with_constraint(N_probs, N_labels, max_fpr=0.4)
    X_optimal_threshold = find_threshold_with_constraint(X_probs, X_labels, max_fpr=0.4)
    optimal_threshold = find_threshold_with_constraint(probs, labels, max_fpr=0.4)
    print("Optimal Threshold (N):", N_optimal_threshold)
    print("Optimal Threshold (X):", X_optimal_threshold)
    print("Optimal Threshold (Final):", optimal_threshold)

    # Compute ROC curve
    N_fpr, N_tpr, N_thresholds = roc_curve(N_labels, N_probs)
    X_fpr, X_tpr, X_thresholds = roc_curve(X_labels, X_probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)

    # Compute AUROC
    N_roc_auc = auc(N_fpr, N_tpr)
    X_roc_auc = auc(X_fpr, X_tpr)
    roc_auc = auc(fpr, tpr)
    print("AUROC (N):", N_roc_auc)
    print("AUROC (X):", X_roc_auc)
    print("AUROC (Final):", roc_auc)

    # Compute metrics at optimal thresholds
    N_metrics = calc_metrics(N_probs, N_labels, N_optimal_threshold)
    X_metrics = calc_metrics(X_probs, X_labels, X_optimal_threshold)
    metrics = calc_metrics(probs, labels, optimal_threshold)
    print("Metrics at Optimal Threshold (N):", N_metrics)
    print("Metrics at Optimal Threshold (X):", X_metrics)
    print("Metrics at Optimal Threshold (Final):", metrics)

    # Plot ROC curve
    plt.figure(figsize=(6,6))
    plt.plot(N_fpr, N_tpr, color='blue', lw=2, label=f'ROC curve using Unexpanded Data (AUC = {N_roc_auc:.2f})')
    plt.plot(X_fpr, X_tpr, color='red', lw=2, label=f'ROC curve using Expanded Data (AUC = {X_roc_auc:.2f})')
    plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve using Expanded Data + Weighted Loss (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

