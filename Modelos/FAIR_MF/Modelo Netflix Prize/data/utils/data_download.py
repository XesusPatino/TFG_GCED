# Downloads the requested dataset DATASET_NAME and saves it in the
# data/DATASET_NAME folder.

# Usage: python download_data.py DATASET_NAME (e.g. python download_data.py ml-100k)

# Supported datasets:
# - ml-100k
# - ml-1m
# - ml-10m

import tarfile
import requests
import zipfile
import io
import sys
import os

DATASETS_DIR = "data/datasets"


def download_and_extract_from_url(url, dataset_name):
    """
    Downloads and extracts the dataset from the given url
    and saves it in the data/dataset_name folder.

    Parameters:
        url (str): The url of the dataset.
        dataset_name (str): The name of the dataset.
    """

    print("Downloading and extracting " + dataset_name + " dataset...")
    r = requests.get(url)

    # If file is a zip file, extract it
    if zipfile.is_zipfile(io.BytesIO(r.content)):
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("data/datasets/")

    # If file is a tar file, extract it
    elif tarfile.is_tarfile(io.BytesIO(r.content)):
        z = tarfile.open(fileobj=io.BytesIO(r.content))
        z.extractall("data/datasets/")
    else:
        # Save the file
        os.makedirs("data/datasets/" + dataset_name, exist_ok=True)
        with open("data/datasets/" + dataset_name + "/temp_data", "wb") as f:
            f.write(r.content)

    print("Done!")


def download_data(dataset_name):
    """
    Calls the download_and_extract_from_url function
    with the correct url for the given dataset_name.

    Parameters:
        dataset_name (str): The name of the dataset.
    """

    # MovieLens 100k
    if dataset_name in ["ml-100k", "ml-1m"]:
        download_and_extract_from_url(
            f"http://files.grouplens.org/datasets/movielens/{dataset_name}.zip",
            dataset_name,
        )

    else:
        print("Dataset not supported yet.")
        return

    print("Dataset " + dataset_name + " downloaded and extracted successfully.")


if __name__ == "__main__":
    # Read dataset name from command line
    if len(sys.argv) != 2:
        print("Usage: python download_data.py DATASET_NAME")
        sys.exit(1)

    dataset_name = sys.argv[1]

    download_data(dataset_name)
