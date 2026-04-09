import numpy as np
from pathlib import Path
import pickle
import gzip
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

BASEDIR = Path(__file__).resolve().parent
REPRESENTATION_PATH = BASEDIR / "Data/Processed/"

id_to_res = {
    4: 'A',
    5: 'R',
    6: 'N',
    7: 'D',
    8: 'C',
    9: 'E',
    10: 'Q',
    11: 'G',
    12: 'H',
    13: 'I',
    14: 'L',
    15: 'K',
    16: 'M',
    17: 'F',
    18: 'P',
    19: 'S',
    20: 'T',
    21: 'W',
    22: 'Y',
    23: 'V'
}

# flatten hidden representations across all samples
def flatten_hidden_representations(path):
    with gzip.open(path / "test_hidden.pkl.gz", "rb") as f:
        hidden = pickle.load(f)
    X = []  # hidden representations
    type_labels = [] # Nb or Ag
    binding_labels = [] # 0 or 1
    residue_ids = [] # amino acid token IDs
    for sample in hidden:
        nb = sample["nb"]      # (len_nb, 768)
        ag = sample["ag"]      # (len_ag, 768)
        y = sample["label"]
        res_ids = sample["res_ids"]

        # Nb residues
        for i in range(len(nb)):
            X.append(nb[i])
            type_labels.append("Nb")
            binding_labels.append(y)
            residue_ids.append(res_ids[i])

        # Ag residues
        for i in range(len(ag)):
            X.append(ag[i])
            type_labels.append("Ag")
            binding_labels.append(y)
            residue_ids.append(res_ids[i])
    return np.array(X), np.array(type_labels), np.array(binding_labels), np.array(residue_ids)

if __name__ == "__main__":
    X, type_labels, binding_labels, residue_ids = flatten_hidden_representations(REPRESENTATION_PATH)
    # print(X.shape)  # Should be (total_residues, 768)
    # print(type_labels.shape)  # Should be (total_residues,)
    # print(binding_labels.shape)  # Should be (total_residues,)
    # print(residue_ids.shape)  # Should be (total_residues,)

    # Convert integer residue IDs to amino acid letters
    res_letters = np.array([id_to_res[i] for i in residue_ids])

    # Perform PCA
    # pca = PCA(n_components=5)
    # X_pca = pca.fit_transform(X)
    # pca5 = PCA(n_components=5)
    # X_pca5 = pca5.fit_transform(X)
    # PC1 = pca5.explained_variance_ratio_[0]
    # PC2 = pca5.explained_variance_ratio_[1]
    # PC3 = pca5.explained_variance_ratio_[2]
    # PC4 = pca5.explained_variance_ratio_[3]
    # PC5 = pca5.explained_variance_ratio_[4]
    # print("PC1:", PC1)
    # print("PC2:", PC2)
    # print("PC3:", PC3)
    # print("PC4:", PC4)
    # print("PC5:", PC5)

    # cor = np.corrcoef(X_pca[:,2], binding_labels)
    # cor4 = np.corrcoef(X_pca[:,3], binding_labels)
    # cor5 = np.corrcoef(X_pca[:,4], binding_labels)
    # print("Correlation between PC3 and binding labels:", cor[0,1])
    # print("Correlation between PC4 and binding labels:", cor4[0,1])
    # print("Correlation between PC5 and binding labels:", cor5[0,1])

    # # Create combined labels (Nb/Ag + binding)
    # combined_labels = [
    #     f"{t}_{y}" for t, y in zip(type_labels, binding_labels)
    # ]

    # # Define color map
    # color_map = {
    #     "Nb_1": "red",      # Nb binding
    #     "Nb_0": "orange",   # Nb non-binding
    #     "Ag_1": "blue",     # Ag binding
    #     "Ag_0": "cyan"      # Ag non-binding
    # }

    # colors = [color_map[label] for label in combined_labels]

    # plt.figure(figsize=(6, 5))

    # plt.scatter(
    #     X_pca[:, 2],
    #     X_pca[:, 3],
    #     c=colors,
    #     alpha=0.3,
    #     s=5
    # )

    # plt.xlabel("PC3")
    # plt.ylabel("PC4")
    # plt.title("Residue PCA: Nb/Ag + Binding")

    # # Add legends
    # for label, color in color_map.items():
    #     plt.scatter([], [], c=color, label=label)

    # plt.legend(markerscale=3)
    # plt.savefig("pc3_pc4_NbAg_Binding.png", dpi=300, bbox_inches="tight")
    # plt.show()

    # # Filter
    # mask = (type_labels == "Ag") & (binding_labels == 1)

    # X_pca_filtered = X_pca[mask]
    # res_letters_filtered = res_letters[mask]

    # # Assign a color for each residue letter
    # unique_res = np.unique(res_letters_filtered)
    # cmap = plt.get_cmap('tab20', len(unique_res))  # up to 20 distinct colors
    # color_dict = {res: cmap(i) for i, res in enumerate(unique_res)}
    # colors = [color_dict[res] for res in res_letters_filtered]

    # # Plot
    # plt.figure(figsize=(6,5))
    # plt.scatter(X_pca_filtered[:,0], X_pca_filtered[:,1], c=colors, alpha=0.5, s=10)
    # plt.title("PCA: Ag Binding Residues")
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")

    # # Add legend
    # for res, color in color_dict.items():
    #     plt.scatter([], [], c=[color], label=res)
    # plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')

    # plt.savefig("pca_Ag_res.png", dpi=300, bbox_inches="tight")
    # plt.show()

    # Perform UMAP
    import umap
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(
        n_components=2, 
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    mask = ((type_labels == "Nb") & (binding_labels == 1)) | ((type_labels == "Ag") & (binding_labels == 1))
    X_umap_filtered = X_umap[mask]
    res_letters_filtered = res_letters[mask]

    combined_labels = [
        f"{t}_{y}" for t, y in zip(type_labels, binding_labels)
    ]

    color_map = {
        "Nb_1": "red",      # Nb binding
        "Nb_0": "orange",   # Nb non-binding
        "Ag_1": "blue",     # Ag binding
        "Ag_0": "cyan"      # Ag non-binding
    }

    colors = [color_map[label] for label in combined_labels]

    # unique_res = np.unique(res_letters_filtered)
    # cmap = plt.get_cmap('tab20', len(unique_res))  # up to 20 distinct colors
    # color_dict = {res: cmap(i) for i, res in enumerate(unique_res)}
    # colors = [color_dict[res] for res in res_letters_filtered]

    plt.figure(figsize=(6,5))
    plt.scatter(X_umap_filtered[:,0], X_umap_filtered[:,1], c=colors, alpha=0.5, s=10)
    plt.title("UMAP: Nb Binding Residues")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")

    # for res, color in color_dict.items():
    #     plt.scatter([], [], c=[color], label=res)
    # plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')

    for label, color in color_map.items():
        plt.scatter([], [], c=color, label=label)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig("umap_Nb/Ag_binding_res.png", dpi=300, bbox_inches="tight")
    # plt.show()