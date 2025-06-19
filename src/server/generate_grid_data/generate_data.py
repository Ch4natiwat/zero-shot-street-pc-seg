from pathlib import Path
from glob import glob
import random as rd
import numpy as np
import torch
import os


GRID_SIZE = 100
VAL_SPLIT = 0.2
SAMPLES_PER_SCENE = 50
MIN_POINTS_PER_SAMPLE = 5
MAX_ITER = SAMPLES_PER_SCENE * 2
    
    
def generate_scene_data(dataroot: str, scene_token: str, split: str="train"):
    
    xyz_path = f"data/output/{split}/xyz"
    embeddings_path = f"data/output/{split}/embeddings"
    Path(xyz_path).mkdir(parents=True, exist_ok=True)
    Path(embeddings_path).mkdir(parents=True, exist_ok=True)
    
    xyz = None
    ego_xyz = None
    embeddings = None
    for data_path in glob(f"{dataroot}/{scene_token}/*.pt"):
        data = torch.load(data_path, weights_only=False)
        coords = data["coords"]
        coords = coords[data["mask_full"]]
        feat = data["feat"]
        ego_translation = data["ego_translation"].unsqueeze(0)
        if xyz is None:
            xyz = coords.clone()
            ego_xyz = ego_translation.clone()
            embeddings = feat.clone()
        else:
            xyz = torch.cat((xyz, coords), dim=0)
            ego_xyz = torch.cat((ego_xyz, ego_translation), dim=0)
            embeddings = torch.cat((embeddings, feat), dim=0)
            
    x_min, x_max = xyz[:, 0].max(), xyz[:, 0].min()
    y_min, y_max = xyz[:, 1].max(), xyz[:, 1].min()
    
    sample_id = 0
    for _ in range(MAX_ITER):
        grid_center_x = rd.uniform(x_min, x_max)
        grid_center_y = rd.uniform(y_min, y_max)
        rotation_angle = rd.uniform(0, 2 * np.pi)
        rot = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle), np.cos(rotation_angle), 0],
            [0, 0, 1]
        ])
        xyz_shifted = xyz.clone()
        xyz_shifted[:, 0] -= grid_center_x
        xyz_shifted[:, 1] -= grid_center_y
        xyz_rotated = xyz_shifted @ rot.T
        xyz_rotated[:, 0] += grid_center_x
        xyz_rotated[:, 1] += grid_center_y
        mask = (
            (grid_center_x - GRID_SIZE // 2 <= xyz_rotated[:, 0]) &
            (grid_center_x + GRID_SIZE // 2 >= xyz_rotated[:, 0]) &
            (grid_center_y - GRID_SIZE // 2 <= xyz_rotated[:, 1]) &
            (grid_center_y + GRID_SIZE // 2 >= xyz_rotated[:, 1])
        )
        if mask.sum() < MIN_POINTS_PER_SAMPLE:
            continue
        
        sample_xyz = xyz_rotated[mask]
        sample_embeddings = embeddings[mask]
        
        np.save(f"{xyz_path}/{scene_token}_{sample_id}.npy", sample_xyz.numpy())
        np.save(f"{embeddings_path}/{scene_token}_{sample_id}.npy", sample_embeddings.numpy())
        
        sample_id += 1
        if sample_id > SAMPLES_PER_SCENE:
            break
        


def generate_data():
    
    rd.seed(42)
    
    dataroot = "data/fusion"
    scene_tokens = os.listdir(dataroot)
    train_size = round(len(scene_tokens) * (1 - VAL_SPLIT))

    rd.shuffle(scene_tokens)
    train_scene_tokens, val_scene_tokens = scene_tokens[:train_size], scene_tokens[train_size:]
    
    for scene_token in train_scene_tokens:
        generate_scene_data(dataroot, scene_token, "train")
        
    for scene_token in val_scene_tokens:
        generate_scene_data(dataroot, scene_token, "val")


if __name__ == "__main__":
    
    generate_data()