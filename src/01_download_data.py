
# -*- coding: utf-8 -*-
"""
01_download_data.py

This is the first script I run in the whole pipeline. All it does is pull
the CK+ (Extended Cohn-Kanade) dataset down from Kaggle and drop it into
data/raw/. I split this into its own file instead of burying it inside the
preprocessing step because I don't want to re-download a few hundred
megabytes every single time I tweak my augmentation logic.

Kaggle switched their API over to a new token format (it now starts with
"KGAT_" and shows up as an "access token" on the settings page, instead
of the old kaggle.json username/key pair). The kagglehub library is the
one that understands this new token natively, so I use that here instead
of the older opendatasets/kaggle.json route.

Before running this I need a Kaggle account and an API token:
  1. Go to kaggle.com -> Settings -> API -> Create New Token
  2. Kaggle shows a token that starts with "KGAT_"
  3. Set it up one of these ways (either is fine, I only need one):

     Option A - environment variable, good for a quick one-off run:
         export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

     Option B - saved token file, good if I don't want to export it
     every time I open a new terminal:
         mkdir -p ~/.kaggle
         echo KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxx > ~/.kaggle/access_token
         chmod 600 ~/.kaggle/access_token

I never put my actual token value inside this script or commit it to
GitHub, it's a secret the same way a password is.

Then just run:
    python src/01_download_data.py
"""

import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, CKPLUS_ROOT, ensure_dirs, refresh_labels

KAGGLE_DATASET = 'shawon10/ckplus'


def check_kaggle_credentials():
    """I check for either valid auth method up front so I can fail with a
    clear message instead of a confusing stack trace halfway through the
    download."""
    has_env_token = bool(os.environ.get('KAGGLE_API_TOKEN'))
    has_token_file = os.path.exists(os.path.expanduser('~/.kaggle/access_token'))
    has_legacy_json = os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json'))

    if has_env_token or has_token_file or has_legacy_json:
        return True

    print('I could not find any Kaggle credentials on this machine.')
    print('Set one of the following before running this script again:')
    print('  1) export KAGGLE_API_TOKEN=KGAT_your_token_here')
    print('  2) save the token to ~/.kaggle/access_token')
    print('  3) (legacy) a kaggle.json file at ~/.kaggle/kaggle.json')
    print('See the docstring at the top of this file for the exact steps.')
    return False


def find_ckplus_folder(search_root):
    """kagglehub caches the download somewhere under its own cache folder
    and I don't want to hardcode that path since it can vary by OS, so I
    just walk the downloaded tree and look for the CK+48 folder by name."""
    for root, dirs, _ in os.walk(search_root):
        if 'CK+48' in dirs:
            return os.path.join(root, 'CK+48')
    return None


def download_ckplus():
    ensure_dirs()

    if os.path.isdir(CKPLUS_ROOT):
        print(f'I already found a dataset at {CKPLUS_ROOT}, skipping download.')
        print('Delete that folder first if I want to force a fresh download.')
        refresh_labels()
        return

    if not check_kaggle_credentials():
        return

    try:
        import kagglehub
    except ImportError:
        print('kagglehub is not installed, run: pip install kagglehub')
        raise

    print(f'Downloading {KAGGLE_DATASET} via kagglehub ...')
    cache_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f'kagglehub cached the dataset at: {cache_path}')

    source_ck48 = find_ckplus_folder(cache_path)
    if source_ck48 is None:
        print('The download finished but I could not find a CK+48 folder')
        print(f'inside {cache_path}. Take a look at that folder manually')
        print('and update CKPLUS_ROOT in config.py if the layout differs.')
        return

    # I copy out of kagglehub's shared cache into my own data/raw/ folder
    # so the rest of the project always reads from a predictable local
    # path instead of depending on kagglehub's cache location.
    dest_ckplus_dir = os.path.dirname(CKPLUS_ROOT)   # data/raw/ckplus
    os.makedirs(dest_ckplus_dir, exist_ok=True)
    print(f'Copying dataset into {CKPLUS_ROOT} ...')
    shutil.copytree(source_ck48, CKPLUS_ROOT)

    if os.path.isdir(CKPLUS_ROOT):
        print('Download finished, dataset is ready at:', CKPLUS_ROOT)
    else:
        print('Something went wrong copying the dataset, check the path above.')

    refresh_labels()


def print_dataset_summary():
    """Quick sanity check so I can see the class folders and how many
    images landed in each one before moving on to preprocessing."""
    if not os.path.isdir(CKPLUS_ROOT):
        print('Dataset folder not found yet, nothing to summarize.')
        return

    labels = sorted(os.listdir(CKPLUS_ROOT))
    print('\nClasses found:', labels)
    total = 0
    for label in labels:
        folder = os.path.join(CKPLUS_ROOT, label)
        if os.path.isdir(folder):
            count = len([f for f in os.listdir(folder)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f'  {label:<12}: {count} images')
            total += count
    print(f'Total: {total} images\n')


if __name__ == '__main__':
    download_ckplus()
    print_dataset_summary()