import pandas as pd
import requests
import os

ROOT_URL = "https://huggingface.co/datasets/m-sakka/agripotential/resolve/main/"
METADATA_URL = ROOT_URL + "metadata.csv"


def download_dataset(dest_dir: str):
    # inside dest_dir, create a agripotential directory
    os.makedirs(os.path.join(dest_dir, "agripotential"), exist_ok=True)
    df_metadata = pd.read_csv(METADATA_URL)
    for idx, df_row in df_metadata.iterrows():
        filename = df_row["filename"]
        file_url = ROOT_URL + filename
        # download this file in dest_dir/agripotential
        print(f"[{idx+1}/{len(df_metadata)}] Downloading {file_url} ...", flush=True)
        dest_path = os.path.join(dest_dir, "agripotential", filename)
        try:
            with requests.get(file_url, stream=True) as response:
                response.raise_for_status()  # Raise exception for HTTP errors
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):  # 1 MB chunks
                        if chunk:
                            f.write(chunk)
            print(f"Downloaded {file_url} successfully.", flush=True)
        except Exception as e:
            print(f"Failed to download {file_url}: {e}", flush=True)
