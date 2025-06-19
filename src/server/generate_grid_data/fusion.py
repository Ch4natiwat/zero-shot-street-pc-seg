from encoder import ImageEncoder, OpenSegImageEncoder
from scipy.spatial.transform import Rotation
from nuscenes_reader import nuScenesReader
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil
import torch
import os        
        

class PointCloudDenseCLIP:
    
    def __init__(
        self, dataroot: str, 
        image_encoder: ImageEncoder, 
        reader_metadata: dict
    ):
        
        self.dataroot = dataroot
        self.image_encoder = image_encoder
        self.image_dim = reader_metadata["image_dim"]
        self.cam_keys = reader_metadata["cam_keys"]
        self.lidar_key = reader_metadata["lidar_key"]
        self.cut_bound = 5


    def compute_mapping(self, camera_to_world, coords, intrinsic):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int)
        inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
                    * (pi[0] < self.image_dim[1] - self.cut_bound) \
                    * (pi[1] < self.image_dim[0] - self.cut_bound)

        front_mask = p[2] > 0
        inside_mask = front_mask * inside_mask
        
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T
    
    
    def get_2d_embeddings(self, samples_data: list, save_dir: str, device="cpu"):
        
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        Path(save_dir).mkdir(parents=True)
        
        def get_transform(sensor_reading):
            
            ego_pose = sensor_reading["ego_pose"]
            sensor_pose = sensor_reading["calibrated_sensor"]
            
            sensor_translation = np.array(sensor_pose["translation"])
            sensor_rotation = Rotation.from_quat(sensor_pose["rotation"][1:] + [sensor_pose["rotation"][0]]).as_matrix()
            ego_translation = np.array(ego_pose["translation"])
            ego_rotation = Rotation.from_quat(ego_pose["rotation"][1:] + [ego_pose["rotation"][0]]).as_matrix()
            
            transform_sensor_to_ego = np.eye(4)
            transform_sensor_to_ego[:3, :3] = sensor_rotation
            transform_sensor_to_ego[:3, 3] = sensor_translation
            transform_ego_to_world = np.eye(4)
            transform_ego_to_world[:3, :3] = ego_rotation
            transform_ego_to_world[:3, 3] = ego_translation
            transform_sensor_to_world = transform_ego_to_world @ transform_sensor_to_ego
            
            return transform_sensor_to_world
        
        def get_coords(lidar_reading):
            
            lidar_to_world = get_transform(lidar_reading)
            
            filename = lidar_reading["filename"]
            path = os.path.join(self.dataroot, filename)
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3]
            
            lidar_hom = np.hstack([points, np.ones((points.shape[0], 1))])
            coords_hom = (lidar_to_world @ lidar_hom.T).T 
            coords = coords_hom[:, :3]
            
            return coords
        
        def get_camera_to_world_intrinsic(cam_reading):
            
            camera_to_world = get_transform(cam_reading)
            intrinsic = np.array(cam_reading["calibrated_sensor"]["camera_intrinsic"])
            
            return camera_to_world, intrinsic
            
        for lidar_frame, sample in enumerate(tqdm(samples_data)):
            lidar_reading = sample[self.lidar_key]
            sample_token = sample["token"]
            coords = get_coords(lidar_reading)
            
            num_points = coords.shape[0]
            frame_visibility_id = torch.zeros((num_points, len(self.cam_keys)), dtype=int, device=device)
            
            ego_translation = torch.FloatTensor(lidar_reading["ego_pose"]["translation"])
            
            counter = torch.zeros((num_points, 1), device=device).half()
            total_features = torch.zeros((num_points, 768), device=device).half()
        
            for i, cam in enumerate(self.cam_keys):
                
                cam_reading = sample[cam]
                ego_translation += torch.FloatTensor(cam_reading["ego_pose"]["translation"])
                camera_to_world, intrinsic = get_camera_to_world_intrinsic(cam_reading)
                mapping = np.ones((num_points, 4), dtype=int)
                
                mapping[:, 1:4] = self.compute_mapping(camera_to_world, coords, intrinsic)

                mapping = torch.from_numpy(mapping).to(device)
                mask = mapping[:, 3]
                frame_visibility_id[:, i] = mask

                image_path = os.path.join(self.dataroot, cam_reading["filename"])
                _, feat_2d = self.image_encoder.encode(image_path)
                
                feat_2d = torch.from_numpy(feat_2d).to(device).permute(2, 0, 1).unsqueeze(0).half()
                feat_2d = F.interpolate(feat_2d, size=self.image_dim, mode="nearest").squeeze()
                    
                feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0).numpy()
                counter[mask != 0] += 1
                total_features[mask != 0] += feat_2d_3d[mask != 0]
            
            counter[counter == 0] = 1e-5
            total_features = total_features / counter
                
            point_ids = torch.unique(frame_visibility_id.nonzero(as_tuple=False)[:, 0])
            mask = torch.zeros(num_points, dtype=torch.bool, device=device)
            mask[point_ids] = True
                
            feat = total_features[mask]
            feat = feat / feat.norm(dim=-1, keepdim=True)
            
            ego_translation /= len(self.cam_keys) + 1
            
            file_path = os.path.join(save_dir, f"{sample_token}.pt")
            torch.save({
                "coords": torch.from_numpy(coords), 
                "ego_translation": ego_translation,
                "mask_full": mask.cpu(), 
                "feat": feat.cpu()
            }, file_path)
    
    
if __name__ == "__main__":
    
    dataroot = os.path.abspath("data/nuscenes/")
    image_model_path = os.path.abspath("exported_model")
    target_path = os.path.abspath("data")
    
    reader = nuScenesReader(dataroot)
    image_encoder = OpenSegImageEncoder(image_model_path)
    model = PointCloudDenseCLIP(dataroot, image_encoder, reader.metadata)
    
    for scene_idx in range(len(reader)):
        reader.set_scene(scene_idx)
        samples_data = reader.get_samples_data()
        scene_token = reader.get_scene_token()
        save_dir = f"{target_path}/fusion/{scene_token}"
        
        model.get_2d_embeddings(samples_data, save_dir, device="cuda")