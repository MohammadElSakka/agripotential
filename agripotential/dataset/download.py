import pandas as pd
import requests
import os

from .urls import ROOT_URL


def download_file(src_url: str, dest_path: str):
    with requests.get(src_url, stream=True) as response:
        response.raise_for_status()  # Raise exception for HTTP errors
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                if chunk:
                    f.write(chunk)


def download_dataset(dest_dir: str):
    # inside dest_dir, create a agripotential directory
    data_dir = os.path.join(dest_dir, "agripotential")

    os.makedirs(data_dir, exist_ok=True)

    # csv: download train.csv, val.csv, metadata.csv,
    # labels (.tif): viticulture.tif, market.tif, field.tif
    # images (.tif): info in metadata.csv
    csv_files = ["train.csv", "val.csv", "metadata.csv"]
    label_files = ["viticulture.tif", "market.tif", "field.tif"]
    metadata_path = f"{data_dir}/metadta.csv"
    df_metadata = pd.read_csv(metadata_path)

    #### csv
    print("Downloading csv files...", flush=True)
    for idx, csv_file in enumerate(csv_files):
        file_url = ROOT_URL + csv_file
        try:
            print(
                f"[{idx}/{len(csv_files)}] Downloading {file_url}...",
                end=" ",
                flush=True,
            )
            download_file(src_url=file_url, dest_path=data_dir)
            print(f"Done.", flush=True)
        except Exception as e:
            print(f"\nFailed to download {file_url}: {e}", flush=True)
            return False

    #### Labels
    print("Downloading label files...", flush=True)
    for idx, label_file in enumerate(label_files):
        file_url = ROOT_URL + label_file
        try:
            print(
                f"[{idx}/{len(label_files)}] Downloading {file_url}...",
                end=" ",
                flush=True,
            )
            download_file(src_url=file_url, dest_path=data_dir)
            print(f"Done.", flush=True)
        except Exception as e:
            print(f"\nFailed to download {file_url}: {e}", flush=True)
            return False

    #### Satellite images
    print("Downloading satellite images...", flush=True)
    for idx, df_row in df_metadata.iterrows():
        filename = df_row["filename"]
        file_url = ROOT_URL + filename
        print(
            f"[{idx+1}/{len(df_metadata)}] Downloading {file_url} ...",
            end=" ",
            flush=True,
        )
        dest_path = os.path.join(dest_dir, "agripotential", filename)
        try:
            download_file(src_url=file_url, dest_path=dest_path)
            print(f"Done.", flush=True)
        except Exception as e:
            print(f"\nFailed to download {file_url}: {e}", flush=True)
            return False

    return True
