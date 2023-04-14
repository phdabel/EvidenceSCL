import sys
import os
import argparse
import zipfile
import validators
import wget


def download(url, target_dir):
    """
    Download a file and save it in some target directory.
    Args:
        url: The url from which the file must be downloaded.
        target_dir: The path to the directory where the file must be saved.
    Returns:
        The path to the downloaded file.
    """
    print("* Downloading data from {}...".format(url))
    filepath = os.path.join(target_dir, url.split('/')[-1])
    wget.download(url, filepath)
    return filepath


def unzip(filepath):
    """
    Extract the data from a zipped file and delete the archive.
    Args:
        filepath: The path to the zipped file.
    """
    print("\n* Extracting: {}...".format(filepath))
    dir_path = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        for name in zf.namelist():
            # Ignore useless files in archives.
            if "__MACOSX" in name or\
               ".DS_Store" in name or\
               "Icon" in name:
                continue
            zf.extract(name, dir_path)
    # Delete the archive once the data has been extracted.
    os.remove(filepath)


def download_unzip(url, target_dir):
    """
    Download and unzip data from some url and save it in a target directory.
    Args:
        url: The url to download the data from.
        target_dir: The target directory in which to download and unzip the
                   data.
    """
    filepath = os.path.join(target_dir, url.split('/')[-1])
    target = os.path.join(target_dir,
                          ".".join((url.split('/')[-1]).split('.')[:-1]))

    if not os.path.exists(target_dir):
        print("* Creating target directory {}...".format(target_dir))
        os.makedirs(target_dir)

    # Skip download and unzipping if the unzipped data is already available.
    if os.path.exists(target) or os.path.exists(target + ".txt"):
        print("* Found unzipped data in {}, skipping download and unzip..."
              .format(target_dir))
    # Skip downloading if the zipped data is already available.
    elif os.path.exists(filepath):
        print("* Found zipped data in {} - skipping download..."
              .format(target_dir))
        unzip(filepath)
    # Download and unzip otherwise.
    else:
        unzip(download(url, target_dir))


if __name__ == "__main__":
    # Default data.
    # Multi-NLI and NLI4CT
    multi_nli_url = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
    nli4ct_url = "https://github.com/ai-systems/nli4ct/raw/main/Complete_dataset.zip"
    parser = argparse.ArgumentParser(description='Download dataset')
    parser.add_argument("--dataset",
                        default='NLI4CT',
                        help="Name of the dataset to download or url of the dataset")
    parser.add_argument("--target_dir",
                        default=os.path.join("../datasets/raw"),
                        help="Path to a directory where raw data must be saved")
    args = parser.parse_args()
    if args.dataset == 'NLI4CT':
        args.dataset_url = nli4ct_url
    elif args.dataset == 'MultiNLI':
        args.dataset_url = multi_nli_url
    elif validators.url(args.dataset):
        args.dataset_url = args.dataset
    else:
        raise Exception("Dataset name not recognized. Please use 'NLI4CT' or 'MultiNLI' or provide a valid url")

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    print(20*"=", "Fetching the dataset:", 20*'=')
    print("Dataset name:", args.dataset)
    download_unzip(args.dataset_url, args.target_dir)

    sys.exit(0)
