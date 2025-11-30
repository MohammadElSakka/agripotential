import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from typing import Tuple, Optional, Literal

from .urls import ROOT_URL


class PotentialDataset:
    def __init__(
        self,
        label_name: Literal["viticulture", "market", "field"],
        mode: Literal["train", "val"],
        data_path: Optional[str] = None,
    ):
        if data_path:
            self.data_path = data_path
            self.metadata_path = os.path.expanduser(f"{self.data_path}/metadata.csv")
            self.patch_csv_path = os.path.expanduser(f"{data_path}/{mode}.csv")
            self.label_path = f"{self.data_path}/{label_name}.tif"
        else:
            self.data_path = ROOT_URL
            self.metadata_path = ROOT_URL + "metadata.csv"
            self.patch_csv_path = ROOT_URL + mode + ".csv"
            self.label_path = ROOT_URL + label_name + ".tif"

        ############# here
        self.sentinel2_paths: list[str] = []
        self.patches: pd.DataFrame = pd.DataFrame()
        self._setup()

    def _setup(self):
        metadata_df = pd.read_csv(self.metadata_path)
        self.sentinel2_paths = [
            f"{self.data_path}/{f}" for f in metadata_df["filename"]
        ]
        self.patch_locations = pd.read_csv(self.patch_location_path)

    def __len__(self) -> int:
        return len(self.patch_locations)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        patch_meta = self.patch_locations.iloc[idx]
        row, col, patch_size = (
            patch_meta["row"],
            patch_meta["col"],
            patch_meta["patch_size"],
        )
        window = Window(col, row, patch_size, patch_size)

        data = np.empty((34, 10, patch_size, patch_size), dtype=np.float32)
        for i, fp in enumerate(self.sentinel2_paths):
            with rasterio.open(fp) as src:
                data[i] = src.read(window=window)

        with rasterio.open(self.labels_map_path) as src:
            label = src.read(window=window)[0].astype(np.int64)

        return data, label
