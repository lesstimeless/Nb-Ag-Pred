import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import biotite.sequence as biotite_seq
import numpy as np
import pandas as pd
import os
from anarci import anarci
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parent

metadataPath = BASE_DIR / "Data/metadata"
pdbPath = BASE_DIR / "Data/pdbs"
output_excel_path = BASE_DIR / "Data/Processed/processed_sequences.xlsx"
imgt_parquet_path = BASE_DIR / "Data/Processed/imgt_aligned_nb.parquet"


def load_metadata(metadataPath):
    """
    Load .tsv metadata files from the given path

    Parameters:
    metadataPath (str): Path to the metadata file

    Returns:
    pd.DataFrame: DataFrame containing 3 columns - pdb, Hchain, antigen_chain
    antigen_chain is a list of chains
    """
    all_metadata = pd.DataFrame()
    for file in os.listdir(metadataPath):
        if file.endswith(".tsv"):
            file_path = os.path.join(metadataPath, file)
            df = pd.read_csv(file_path, sep="\t", dtype={'pdb': str})

            # Ensure pdb column is string type
            id_col = df['pdb'].astype(str)

            if df['Hchain'].dtype == object or df['Hchain'].dtype != int:
                Nb_col = df['Hchain']
            else:
                Nb_col = df['Hchain'].apply(lambda x: str(x))
            if df['antigen_chain'].dtype == object:
                Ag_col = df['antigen_chain'].str.split('|')
            elif df['antigen_chain'].dtype == int:
                Ag_col = df['antigen_chain'].apply(lambda x: [str(x)])
            else:
                Ag_col = df['antigen_chain']
            metadata = pd.concat([id_col, Nb_col, Ag_col], axis=1)
            all_metadata = pd.concat([all_metadata, metadata], axis=0)
    return all_metadata
            
# To convert atom-level coordinates to residue-level coordinates
def group_consecutive_mean_numpy(A, B):
    # Find where labels change
    change_points = np.where(B[:-1] != B[1:])[0] + 1
    group_starts = np.concatenate(([0], change_points))
    group_ends = np.concatenate((change_points, [len(B)]))
    
    means = []
    labels = []
    sizes = []
    indices = []
    
    for start, end in zip(group_starts, group_ends):
        group_coords = A[start:end]
        means.append(np.mean(group_coords, axis=0))
        labels.append(B[start])
        sizes.append(end - start)
        indices.append((start, end-1))
    
    return {
        'labels': np.array(labels),
        'means': np.array(means),
        'sizes': np.array(sizes),
        'indices': indices
    }

# To identify binding residues based on distance threshold
# Used in get_sequences_from_pdb function
def binding_residues(Nb_struc, Ag_struc, Nb_sequence, Ag_sequence, threshold=6.0):
    """
    Identify binding residues between nanobody and antigen based on distance threshold

    Parameters:
    Nb_struc (biotite.structure): Structure of the nanobody
    Ag_struc (biotite.structure): Structure of the antigen
    threshold (float): Distance threshold to consider a residue as binding

    Returns:
    tuple: Two lists containing indices of binding residues in Nb_struc and Ag_struc
    """

    Nb_atm_coords = struc.coord(Nb_struc)
    Ag_atm_coords = struc.coord(Ag_struc)

    Nb_res_labels = struc.create_continuous_res_ids(Nb_struc)
    Ag_res_labels = struc.create_continuous_res_ids(Ag_struc)

    # print(len(Nb_sequence), len(Nb_res_labels), len(Ag_sequence), len(Ag_res_labels))
    # print(max(Nb_res_labels), len(Nb_sequence))

    if max(Nb_res_labels) != len(Nb_sequence):
        print(f"In binding_residues func, Nb: {max(Nb_res_labels)}, {len(Nb_sequence)}, {struc.get_residue_count(Nb_struc)}")
    if max(Ag_res_labels) != len(Ag_sequence):
        print(f"In binding_residues func, Ag: {max(Ag_res_labels)}, {len(Ag_sequence)}, {struc.get_residue_count(Ag_struc)}")
        print(len(Ag_res_labels), len(Ag_atm_coords))

    Nb_res_coords = group_consecutive_mean_numpy(Nb_atm_coords, Nb_res_labels)['means']
    Ag_res_coords = group_consecutive_mean_numpy(Ag_atm_coords, Ag_res_labels)['means']

    diff = Nb_res_coords[:, np.newaxis, :] - Ag_res_coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    
    binding_pairs = np.where(distances < threshold)

    Nb_binding_residues = binding_pairs[0]
    Ag_binding_residues = binding_pairs[1]

    Nb_binding_residues = tuple(set(Nb_binding_residues.astype(int).tolist()))
    Ag_binding_residues = tuple(set(Ag_binding_residues.astype(int).tolist()))

    return Nb_binding_residues, Ag_binding_residues


# Extract sequences and binding residues from a PDB file
## Used in process_all_pdbs function
def get_sequences_from_pdb(pdb_file, Nb_chain: str, Ag_chains: list):
    """
    Extract sequences from a PDB file
    
    Parameters:
    pdb_file (str): Path to PDB file
    Nb_chain (str): Chain ID of the nanobody
    Ag_chains (list): List of chain IDs of the antigens
    
    Returns:
    pd.DataFrame: DataFrame with 12 columns - pdb, Nb_sequence, Ag_sequence, Nb_binding_idx, Ag_binding_idx, Nb_mask_idx, Ag_mask_idx, FR1, FR2, FR3, FR4, imgt_binding
    """
    sequences_df = pd.DataFrame(columns=['pdb', 'Nb_sequence', 'Ag_sequence', 'Nb_binding_idx', 'Ag_binding_idx', 
                                         'CDR_indices', 'four_FR_indices', 'rest_FR_indices',
                                         'FR1', 'FR2', 'FR3', 'FR4',
                                         'imgt_binding'])

    # Load structure
    # print(pdb_file)
    try:
        raw_structure = strucio.load_structure(pdb_file)
    except Exception as e:
        print(f"Error loading {pdb_file}: {e}")
        return None
    # Remove hetero atoms and non-standard residues
    structure = raw_structure[
        (raw_structure.hetero == False) & 
        struc.filter_amino_acids(raw_structure)
    ]

    # Extract Nb sequence
    Nb_struc = structure[structure.chain_id == Nb_chain]
    dic = {"FR1": [], "FR2": [], "FR3": [], "FR4": [], "CDR1": [], "CDR2": [], "CDR3": []}
    Nb_mask_idx = []
    four_res_idx = []
    rest_fr_idx = []
    cdr_idx = []
    Nb_sequence = []
    imgt_binding = []

    if len(Nb_struc) > 0:
        temp_Nb_sequence = struc.get_residues(Nb_struc)[1]
        for ele in temp_Nb_sequence:
            Nb_sequence.append(biotite_seq.ProteinSequence.convert_letter_3to1(ele))
        Nb_sequence = "".join(Nb_sequence)

        # Extract Ag sequences
        Ag_sequences = []
        binding_idx = []

        for Ag_chain in Ag_chains:
            Ag_struc = structure[structure.chain_id == Ag_chain.strip()]
            Ag_sequence = []
            temp_Ag_idx = []
            if len(Ag_struc) > 0:
                temp_Ag_sequence = struc.get_residues(Ag_struc)[1]
                for idx, ele in enumerate(temp_Ag_sequence):
                    Ag_sequence.append(biotite_seq.ProteinSequence.convert_letter_3to1(ele))
                    temp_Ag_idx.append(idx)
                Ag_sequence = "".join(Ag_sequence)

                # Extract binding residues of Nb and Ag
                Nb_binding_idx, Ag_binding_idx = binding_residues(Nb_struc, Ag_struc, Nb_sequence, Ag_sequence)
                if Ag_binding_idx:
                    if max(Ag_binding_idx) >= len(Ag_sequence):
                        print(f"Warning: Ag Binding index out of range in {pdb_file} for chain {Ag_chain}")
                        print(Ag_binding_idx, len(Ag_sequence))
                        continue
                if Nb_binding_idx:
                    if max(Nb_binding_idx) >= len(Nb_sequence):
                        print(f"Warning: Nb Binding index out of range in {pdb_file} for chain {Nb_chain}")
                        print(Nb_binding_idx, len(Nb_sequence))
                        continue
                if Nb_binding_idx and Ag_binding_idx:
                    binding_idx.append((Nb_binding_idx, Ag_binding_idx))
                else:
                    binding_idx.append(((), ()))
                Ag_sequences.append(Ag_sequence)
            else:
                print(f"Warning: No antigen chain {Ag_chain} found in {pdb_file}")
                return None
    else:
        print(structure.chain_id, Nb_chain)
        print(f"Warning: No nanobody chain {Nb_chain} found in {pdb_file}")
        return None

    # IMGT alignment
    seq = [(pdb_file, Nb_sequence)]
    Nb_res = anarci(sequences=seq, scheme="imgt")
    temp_insertion = False
    if Nb_res[0][0]:
        imgt = Nb_res[0][0][0][0]
        emp_counter = 0
        trunc_counter = 0

        for idx in range(len(imgt)):
            num = imgt[idx][0][0]
            if imgt[idx][0][1] != ' ':
                insertion = True
                temp_insertion = True
            else:
                insertion = False
            aa = imgt[idx][1]
            actual_idx = idx - emp_counter + trunc_counter
            if aa == "-":
                emp_counter += 1
            # Run the print statements in this else clause to verify if residues position between IMGT-aligned and original sequence matches
            # When position matches --> prints True --> binding residue index corresponds to correct IMGT numbered residue
            else:
                if Nb_sequence[actual_idx] != aa:
                    # print(f"{Nb_sequence[actual_idx]}{actual_idx},{aa}{num}")
                    # print(emp_counter)
                    # print(Nb_sequence)
                    # print(imgt)
                    trunc_counter += 1
                    actual_idx += 1
            if binding_idx and aa != "-":
                for Nb, Ag in binding_idx:
                    if actual_idx in Nb:
                        imgt_binding.append(num)
            if num <= 26:
                if insertion:
                    if type(dic['FR1'][-1]) != list:
                        dic['FR1'][-1] = list(dic['FR1'][-1])
                    dic['FR1'][-1].append(aa)
                else: dic['FR1'].append(aa)
                if aa != "-" and num == 1:
                    four_res_idx.append(actual_idx)
                elif aa != "-" and num != 1:
                    rest_fr_idx.append(actual_idx)
            elif num > 26 and num <= 38:
                if insertion:
                    if type(dic['CDR1'][-1]) != list:
                        dic['CDR1'][-1] = list(dic['CDR1'][-1])
                    dic['CDR1'][-1].append(aa)
                else: dic['CDR1'].append(aa)
                if aa != "-":
                    cdr_idx.append(actual_idx)
            elif num > 38 and num <= 55:
                if insertion:
                    if type(dic['FR2'][-1]) != list:
                        dic['FR2'][-1] = list(dic['FR2'][-1])
                    dic['FR2'][-1].append(aa)
                else: dic['FR2'].append(aa)
                if aa != "-" and num == 52:
                    four_res_idx.append(actual_idx)
                elif aa != "-" and num != 52:
                    rest_fr_idx.append(actual_idx)
            elif num > 55 and num <= 65:
                if insertion:
                    if type(dic['CDR2'][-1]) != list:
                        dic['CDR2'][-1] = list(dic['CDR2'][-1])
                    dic['CDR2'][-1].append(aa)
                else: dic['CDR2'].append(aa)
                if aa != "-":
                    cdr_idx.append(actual_idx)
            elif num > 65 and num <= 104:
                if insertion:
                    if type(dic['FR3'][-1]) != list:
                        dic['FR3'][-1] = list(dic['FR3'][-1])
                    dic['FR3'][-1].append(aa)
                else: dic['FR3'].append(aa)
                if aa != "-" and (num == 66 or num == 69):
                    four_res_idx.append(actual_idx)
                elif aa != "-" and (num != 66 and num != 69):
                    rest_fr_idx.append(actual_idx)
            elif num > 104 and num <= 117:
                if insertion:
                    if type(dic['CDR3'][-1]) != list:
                        dic['CDR3'][-1] = list(dic['CDR3'][-1])
                    dic['CDR3'][-1].append(aa)
                else: dic['CDR3'].append(aa)
                if aa != "-":
                    cdr_idx.append(actual_idx)
            elif num > 117:
                if insertion:
                    if type(dic['FR4'][-1]) != list:
                        dic['FR4'][-1] = list(dic['FR4'][-1])
                    dic['FR4'][-1].append(aa)
                else: dic['FR4'].append(aa)
                if aa != "-":
                    rest_fr_idx.append(actual_idx)
            else:
                print("IMGT indexing error")
                break
    else:
        imgt = None
        dic['FR1'] = Nb_sequence
    imgt_binding.sort()

    # Mask 40% CDR idx, 25% of 4 FR residues, 6% rest of FR residues
    cdr_idx = list(set(cdr_idx))
    four_res_idx = list(set(four_res_idx))
    rest_fr_idx = list(set(rest_fr_idx))

    # print("Mask idx:", mask_idx)
    # print(len(Nb_mask_idx)/len(Nb_sequence), Nb_mask_idx)

    # For each Nb-Ag pair, total combinations are Nb-Ag, Nb-flip(Ag), flip(Nb)-Ag, flip(Nb)-flip(Ag)
    def flip_sequence(seq, binding_idx, cdr_idx, four_res_idx, rest_Fr_idx):
        flipped_seq = seq[::-1]
        length = len(seq)
        flipped_binding = tuple([length - 1 - idx for idx in binding_idx])
        flipped_cdr = [length - 1 - idx for idx in cdr_idx]
        flipped_four_res = [length - 1 - idx for idx in four_res_idx]
        flipped_rest_fr = [length - 1 - idx for idx in rest_Fr_idx]
        return flipped_seq, flipped_binding, flipped_cdr, flipped_four_res, flipped_rest_fr
    
    # Add each Nb-Ag pair to the DataFrame
    for Ag_sequence, (Nb_binding_idx, Ag_binding_idx) in zip(Ag_sequences, binding_idx):
        flip_Nb = flip_sequence(Nb_sequence, Nb_binding_idx, cdr_idx, four_res_idx, rest_fr_idx)
        orientations = [
            (Nb_sequence, Ag_sequence, Nb_binding_idx, Ag_binding_idx, cdr_idx, four_res_idx, rest_fr_idx),
            (Nb_sequence, Ag_sequence[::-1], Nb_binding_idx, tuple([len(Ag_sequence) - 1 - idx for idx in Ag_binding_idx]), cdr_idx, four_res_idx, rest_fr_idx),
            (flip_Nb[0], Ag_sequence, flip_Nb[1], Ag_binding_idx, flip_Nb[2], flip_Nb[3], flip_Nb[4]),
            (flip_Nb[0], Ag_sequence[::-1], flip_Nb[1], tuple([len(Ag_sequence) - 1 - idx for idx in Ag_binding_idx]), flip_Nb[2], flip_Nb[3], flip_Nb[4])
        ]
        for orient in orientations:
            sequences_df.loc[len(sequences_df)] = ({
                'pdb': os.path.basename(pdb_file).replace(".pdb", ""),
                'Nb_sequence': orient[0],
                'Ag_sequence': orient[1],
                'Nb_binding_idx': orient[2],
                'Ag_binding_idx': orient[3],
                'CDR_indices': orient[4],
                'four_FR_indices': orient[5],
                'rest_FR_indices': orient[6],
                'FR1': dic['FR1'],
                'FR2': dic['FR2'],
                'FR3': dic['FR3'],
                'FR4': dic['FR4'],
                'imgt_binding': imgt_binding
            })

    return sequences_df

# Process all PDB files based on the metadata and extract sequences
def process_all_pdbs(metadata, pdbPath):
    """
    Process all PDB files based on the metadata

    Parameters:
    metadata (pd.DataFrame): DataFrame containing pdb, Hchain, antigen_chain
    pdbPath (str): Path to the directory containing PDB files

    Returns:
    pd.DataFrame: DataFrame with all extracted sequences
    --> 12 columns: pdb, Nb_sequence, Ag_sequence, Nb_binding_idx, Ag_binding_idx, CDR_indices, four_FR_indices, rest_FR_indices, FR1, FR2, FR3, FR4, imgt_binding
    """
    all_sequences = pd.DataFrame(columns=['pdb', 'Nb_sequence', 'Ag_sequence', 'Nb_binding_idx', 'Ag_binding_idx', 
                                          'CDR_indices', 'four_FR_indices', 'rest_FR_indices',
                                          'FR1', 'FR2', 'FR3', 'FR4',
                                          'imgt_binding'])
    
    for index, row in metadata.iterrows():
        pdb_id = row['pdb']
        Nb_chain = row['Hchain']
        Ag_chains = row['antigen_chain']

        if not isinstance(Ag_chains, list) and Nb_chain != "nan":
            print(Ag_chains)
            continue
        
        pdb_file = os.path.join(pdbPath, f"{pdb_id}.pdb")
        if os.path.exists(pdb_file):
            sequences_df = get_sequences_from_pdb(pdb_file, Nb_chain, Ag_chains)
            if sequences_df is not None:
                all_sequences = pd.concat([all_sequences, sequences_df], axis=0, ignore_index=True)
        else:
            print(f"Warning: PDB file {pdb_file} does not exist.")

    return all_sequences

# Upload processed data to excel sheet
def upload_to_excel(dataframe, output_path):
    """
    Upload the DataFrame to an Excel file

    Parameters:
    dataframe (pd.DataFrame): DataFrame to be saved
    output_path (str): Path to save the Excel file
    """
    df_unique = dataframe.drop_duplicates(
        subset=['Nb_sequence', 'Ag_sequence', 'Nb_binding_idx', 'Ag_binding_idx'],
        keep='first'
    ).reset_index(drop=True)
    df_negative = df_unique[df_unique['Nb_binding_idx'].apply(len) < 3]
    df_positive = df_unique[df_unique['Nb_binding_idx'].apply(len) >= 3]
    print(f"Number of negative samples: {len(df_negative)}")
    print(f"Number of positive samples: {len(df_positive)}")

    with pd.ExcelWriter(output_path) as writer:
        df_negative.to_excel(writer, sheet_name='Negative Samples', index=False)
        df_positive.to_excel(writer, sheet_name='Positive Samples', index=False)
    print(f"Data successfully saved to {output_path}")
    print("Preprocessing done!")

if __name__ == "__main__":
    # Usage
    metadata_df = load_metadata(metadataPath)
    sequence_df = process_all_pdbs(metadata_df, pdbPath)

    # # Extract unique Nb sequences (IMGT aligned)
    # unique_nb = sequence_df[["pdb", "Nb_sequence", "FR1", "FR2", "FR3", "FR4",
    #                          "imgt_binding"]]
    # unique_nb = unique_nb.drop_duplicates(
    #     subset=["Nb_sequence"],
    #     keep="first"
    # )
    # print(len(unique_nb))
    # unique_nb.to_parquet(imgt_parquet_path)
    # Uncomment when you want to save the output to an Excel file
    upload_to_excel(sequence_df, output_excel_path)
