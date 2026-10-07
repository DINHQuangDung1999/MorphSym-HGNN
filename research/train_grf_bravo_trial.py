"""Run a short 3D GRF training trial using only the A1 Bravo dataset."""

from pathlib import Path

import numpy as np
import torch

from ms_hgnn.datasets_py.quadSDKDataset_Morph import QuadSDKDataset_A1_Bravo
from ms_hgnn.lightning_py.gnnLightning import train_model


def main():
    repo = Path(__file__).resolve().parents[1]
    root = repo.parent / "datasets" / "QuadSDK-A1-Bravo"
    dataset = QuadSDKDataset_A1_Bravo(
        root,
        repo / "urdf_files" / "A1-Quad" / "a1_pruned.urdf",
        "package://a1_description/",
        "",
        "heterogeneous_gnn_c2",
        150,
        False,
        repo / "urdf_files" / "A1-Quad" / "a1.urdf",
        None,
        "MorphSym",
        str(repo / "cfg" / "a1-c2.yaml"),
        True,
        3,
    )

    split = int(np.round((len(dataset) - 1) * 0.85))
    train = torch.utils.data.Subset(dataset, np.arange(split))
    val = torch.utils.data.Subset(dataset, np.arange(split, len(dataset) - 1))
    print(f"Bravo samples: {len(train)} train, {len(val)} validation", flush=True)

    checkpoint_dir = train_model(
        train,
        val,
        None,
        False,
        testing_mode=True,
        disable_logger=True,
        batch_size=8,
        num_layers=8,
        hidden_size=128,
        lr=1e-4,
        epochs=100,
        regression=True,
        seed=0,
        devices=1,
        disable_test=True,
        symmetry_mode="MorphSym",
        group_operator_path=str(repo / "cfg" / "a1-c2.yaml"),
        grf_body_to_world_frame=True,
        grf_dimension=3,
    )
    print(f"Checkpoints: {checkpoint_dir}", flush=True)


if __name__ == "__main__":
    main()
