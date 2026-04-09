import os
import requests
from bs4 import BeautifulSoup
import urllib.request as req

sabdab_nanobody_url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/search/?ABtype=VHH&method=All&species=All&resolution=&rfactor=&antigen=Protein&ltype=All&constantregion=All&affinity=All&isin_covabdab=All&isin_therasabdab=All&chothiapos=&restype=ALA&field_0=Antigens&keyword_0=#downloads"
source_url = "https://opig.stats.ox.ac.uk/"
metadata_output_path = f"{os.getcwd()}/Data/metadata/"
pdb_output_path = f"{os.getcwd()}/Data/pdbs/"


def download_sabdab_metadata(download_dir, url, source):
    """
    Download nanobody summary TSV files from the SAbDab database.

    Parameters:
    download_dir (str): Directory to save the downloaded summary files
    url (str): URL of the SAbDab nanobody (or other database) search results page )
    source (str): Base URL to prepend to relative links
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links with "Summary file"
    links = soup.find_all('a', href=True, string="Summary file")

    for link in links:
        file_url = f"{source}{link['href']}"
        file_name = f"{file_url.split('/')[-2]}.tsv"
        file_path = os.path.join(download_dir, file_name)
        if not os.path.exists(file_path):
            req.urlretrieve(file_url, file_path)
            print(f"Downloaded: {file_name}")
        else:
            print(f"File already exists: {file_name}")

def download_sabdab_pdbs(download_dir, url,source):
    """
    Download PDB files from the SAbDab database.

    Parameters:
    download_dir (str): Directory to save the downloaded PDB files
    url (str): URL of the SAbDab nanobody (or other database) search results page
    source (str): Base URL to prepend to relative links
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links with "PDB file"
    links = soup.find_all('a', href=True, string="Structure (IMGT)")

    for link in links:
        file_url = f"{source}{link['href']}"
        file_name = f"{file_url.split('/')[-2]}.pdb"
        file_path = os.path.join(download_dir, file_name)
        if not os.path.exists(file_path):
            req.urlretrieve(file_url, file_path)
            print(f"Downloaded: {file_name}")
        else:
            print(f"File already exists: {file_name}")

## Download metadata and PDB files, run once
# download_sabdab_metadata(metadata_output_path, sabdab_nanobody_url, source_url)
# download_sabdab_pdbs(pdb_output_path, sabdab_nanobody_url, source_url)

## Run these to check if the number of downloaded files match
# print(f"Number of metadata files = Number of PDB files: {len(os.listdir(metadata_output_path)) == len(os.listdir(pdb_output_path))}")
# print(len(os.listdir(metadata_output_path)), "metadata files found in", metadata_output_path)
# print(len(os.listdir(pdb_output_path)), "PDB files found in", pdb_output_path)
