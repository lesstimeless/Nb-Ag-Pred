import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast
import re

BASE_DIR = Path(__file__).resolve().parent
pd.set_option("display.max_columns", None)

excel_path = BASE_DIR / "Data/Processed/processed_sequences.xlsx"
fr_df = pd.read_excel(excel_path, sheet_name="Positive Samples")

color_dict = {
    "-": "#000000", "A": "#1f77b4", "C": "#ff7f0e", "D": "#2ca02c", "E": "#d62728",
    "F": "#9467bd", "G": "#cb9d94", "H": "#e377c2", "I": "#7f7f7f", "K": "#bcbd22",
    "L": "#17becf", "M": "#393b79", "N": "#637939", "P": "#8c6d31", "Q": "#843c39",
    "R": "#b487ae", "S": "#5254a3", "T": "#9c9ede", "V": "#8ca252", "W": "#ad494a", 
    "Y": "#e7969c"
}
imgt_start = {"FR1": 1, "FR2": 39, "FR3": 66, "FR4": 118, 
              "CDR1": 27, "CDR2":56, "CDR3":105}

def try_literal_eval(val):
    try:
        # Only convert if it's a list literal
        result = ast.literal_eval(val)
        if isinstance(result, list):
            return result
        else:
            return val  # keep as string if not a list
    except (ValueError, SyntaxError):
        return val  # keep as string if eval fails

# Convert string representation to lists
for fr in ["FR1", "FR2", "FR3", "FR4"]:
    fr_df[fr] = fr_df[fr].apply(try_literal_eval)
    fr_df[fr] = fr_df[fr][fr_df[fr].apply(lambda x: isinstance(x, list))]
    fr_df = fr_df[fr_df[fr].apply(lambda x: len(x) > 0)]
fr_df["imgt_binding"] = fr_df["imgt_binding"].apply(try_literal_eval)
fr_df = fr_df[fr_df["imgt_binding"].apply(lambda x: len(x) > 0)]

# Flatten the list and map to its imgt number
def row_to_pos_dic(seq):
    pos = 0
    result = {}
    for el in seq:
        pos += 1
        if isinstance(el, list):
            result[str(pos)] = el[0]
            for i, ins in enumerate(el[1:]):
                suffix = chr(97+i) # a, b, c, ...
                result[f"{pos}{suffix}"] = ins
        else:
            result[str(pos)] = el
    return result

# Sort the columns in ascending imgt numbers
def col_key(col):
    res = re.match(r"(\d+)([a-z]*)", col)
    return (int(res.group(1)), res.group(2))

# generate stacked histogram for residue composition on FRs
def view_fr_composition():
    for fr in ["FR1", "FR2", "FR3", "FR4"]:
        # Flatten each sequence
        expanded = fr_df[fr].apply(row_to_pos_dic)
        max_len = expanded.apply(len).max()
            
        df = pd.DataFrame(expanded.to_list())
        df_aligned = df[sorted(df.columns, key=col_key)]

        # Calculate percentage per residue per position
        freq = df_aligned.apply(
            lambda col: col.value_counts(normalize=True) * 100
        ).fillna(0)
        # print(freq)

        # Assign colors (add fallback for prefixes like A, B, etc.)
        residues = freq.index.unique()
        colors = [color_dict.get(r[-1] if r[0].isalpha() else r, "#808080") for r in residues]

        # Plot stacked histogram
        ax = freq.T.plot(
            kind="bar", 
            stacked=True, 
            figsize=(14, 4), 
            width=0.7, 
            color=colors)

        # Set x-axis labels with IMGT positions and sample sizes
        offset = imgt_start[fr] - 1
        sample_size = df_aligned.notna().sum(axis=0)
        new_labels = [f"{int(col)+offset}\nn=\n{sample_size[col]}" if col.isnumeric() else 
                    f"{int(col[:-1])+offset}{col[-1]}n={sample_size[col]}"
                    for col in df_aligned.columns]
        ax.set_xticks(range(len(df_aligned.columns)))
        ax.set_xticklabels(new_labels, rotation=0)

        for i, position in enumerate(freq.columns): 
            col = freq[position] 
            col = col[col > 0].sort_values(ascending=True) 
            y_start = 107 
            line_spacing = 7 
            for j, (res, val) in enumerate(col.items()): 
                label = f'"{res}" - {val:.2f}' if res == "-" else f"{res} - {val:.2f}" 
                ax.text(
                    i, 
                    y_start + j * line_spacing, 
                    label, ha="center", va="bottom", 
                    fontsize=9, 
                    fontweight="bold" if val > 10 else "normal", 
                    clip_on=False 
                    )

        plt.suptitle(f"{fr} residue composition (IMGT aligned)")
        plt.xlabel("IMGT position")
        plt.ylabel("Residue percentage (%)")
        plt.legend(title="Residue", bbox_to_anchor=(0.5, -0.6), loc="upper center", ncol=11, fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.5, top=0.70)
        plt.show()

# generate binding residue dsitribution on IMGT aligned Nb sequence (full seq)
def view_binding_dist():
    df = fr_df["imgt_binding"]
    exploded_df = df.explode().dropna().astype(int)
    binding_counts = exploded_df.value_counts().sort_index()
    # print(exploded_df)
    print(binding_counts)

    imgt_pos = np.arange(1,129)
    binding_counts = binding_counts.reindex(imgt_pos, fill_value=0)
    values = binding_counts.values
    print(values)

    heat_data = values.reshape(1, -1)
    fig, ax = plt.subplots(figsize=(16, 0.7))

    im = ax.imshow(
        heat_data,
        aspect="auto",
        cmap="hot",
        interpolation="nearest"
    )

    # X-axis = IMGT positions
    ax.set_xticks(range(128))
    ax.set_xticklabels(range(1, 129), fontsize=7)

    # Hide y-axis
    ax.set_yticks([])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05)
    cbar.set_label("Binding occurrence")

    plt.title("Binding residue distribution across IMGT positions")
    plt.xlabel("IMGT position")
    plt.tight_layout()
    plt.show()

# Function to get number of binding residues in Nb and Ag and write to excel
def binding_res_counts():
    Nb = fr_df["Nb_binding_idx"].apply(lambda x: len(ast.literal_eval(x)))
    Ag = fr_df["Ag_binding_idx"].apply(lambda x: len(ast.literal_eval(x)))
    fr_df["Nb_binding_count"] = Nb
    fr_df["Ag_binding_count"] = Ag
    
    # Upload to existing excel without changing other sheets
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        fr_df.to_excel(writer, sheet_name="Positive Samples", index=False)
    
    print("Binding residue counts added to excel.")

if __name__ == "__main__":
    print("Running Commands for Data Visulalization")
    # view_binding_dist()
    binding_res_counts()


## Observation
# 1. Binding Residues in FRs (4 main positions)
# -- FR1: pos 1, FR2: pos 52, FR3: pos 66, 69, FR4: NIL

# 2. High diversity positions in FRs
# -- FR1, pos 1: 63.71% Q, 9.13% E, 2.03% D, 1.05% A
# -- FR2, pos 52: 33.53% W, 29.09% F, 15.63% L, 14.48% G, 1.5% M, 1.22% R
# -- FR3, pos 66: 29.3% Y, 23.04% S, 17.13% N, 5.14% D, 4.27% T, 2.97% H, 2.73% R, 2.62% A, 2.31% K, 2.03% L, 1.71% V, 1.71% F, 1.29% I
# -- FR3, pos 69: 73.15% D, 17.83% G, 2.27% E, 2.2% N, 1.36% P
