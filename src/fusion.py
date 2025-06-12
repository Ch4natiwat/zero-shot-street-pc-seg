from encoder import ImageEncoder, OpenSegImageEncoder, CLIPTextEncoder
from scipy.spatial.transform import Rotation
from nuscenes_reader import nuScenesReader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import open3d
import torch
import os


class ImageEmbeddingsQueue:
    
    def __init__(self, max_length: int):
        self.count = 0
        self.max_length = max_length
        self.queue = None
        self.permutation = tuple(range(max_length))
        self.next_idx = 0
        self.frame_idx_to_queue_idx = {}
        
        
    def reset(self):
        self.__init__(self.max_length)
        
        
    def add(self, image_embeddings: np.ndarray):
        if self.count == 0:
            self.queue = np.zeros((self.max_length, *image_embeddings.shape), dtype=np.float32)
        self.queue[self.next_idx] = image_embeddings
        self.frame_idx_to_queue_idx[self.count] = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.max_length
        self.count += 1
        if self.count > self.max_length:
            self.frame_idx_to_queue_idx.pop(self.count - self.max_length - 1)
        
        
    def get(self, frame_idx: int):
        if frame_idx not in self.frame_idx_to_queue_idx:
            return None
        return self.queue[self.frame_idx_to_queue_idx[frame_idx]]
        
        

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
    
    
    def get_2d_embeddings(
        self, samples_data: list[dict], device: str="cpu", verbose: bool=False,
        do_cache: bool=False, neighbor_frame_weights: list[float]=[1.0] 
    ):
        
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
        
        cam_frames_per_lidar_frame = len(neighbor_frame_weights)
        
        if do_cache:
            cache = ImageEmbeddingsQueue(max_length=cam_frames_per_lidar_frame)
            
        for lidar_frame, sample in enumerate(samples_data[:1]):
            lidar_reading = sample[self.lidar_key]
            coords = get_coords(lidar_reading)
            
            num_points = coords.shape[0]
            
            first_requesting_cam_frame = lidar_frame - (cam_frames_per_lidar_frame // 2)
            requesting_cam_frames = [first_requesting_cam_frame + offset for offset in range(cam_frames_per_lidar_frame)]
            
            total_features = torch.zeros((num_points, 768), device=device)
            frame_visibility_id = torch.zeros((num_points, len(self.cam_keys)), dtype=int, device=device)
            
            for frame_index, cam_frame in enumerate(requesting_cam_frames):
                
                if cam_frame < 0 or cam_frame >= len(samples_data):
                    continue
                
                weight = neighbor_frame_weights[frame_index]
                counter = torch.zeros((num_points, 1), device=device)
                total_frame_features = torch.zeros((num_points, 768), device=device)
                
                if do_cache:
                    cached_feat_2ds = cache.get(cam_frame)
                    cache_hit = cached_feat_2ds is not None
                    feat_2ds = []
            
                for i, cam in enumerate(self.cam_keys):
                    
                    cam_reading = sample[cam]
                    camera_to_world, intrinsic = get_camera_to_world_intrinsic(cam_reading)
                    mapping = np.ones((num_points, 4), dtype=int)
                    
                    mapping[:, 1:4] = self.compute_mapping(camera_to_world, coords, intrinsic)

                    mapping = torch.from_numpy(mapping).to(device)
                    mask = mapping[:, 3]
                    frame_visibility_id[:, i] |= mask
                    
                    if do_cache and cache_hit:
                        feat_2d = cached_feat_2ds[i]
                    else:
                        image_path = os.path.join(self.dataroot, cam_reading["filename"])
                        _, feat_2d = self.image_encoder.encode(image_path)
                        
                        feat_2d = torch.from_numpy(feat_2d).permute(2, 0, 1).unsqueeze(0)
                        feat_2d = F.interpolate(feat_2d, size=self.image_dim, mode="nearest").squeeze()
                        if do_cache:
                            feat_2ds.append(feat_2d.numpy())
                        
                    feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0).numpy()
                    counter[mask != 0] += 1
                    total_frame_features[mask != 0] += feat_2d_3d[mask != 0]
                
                counter[counter == 0] = 1e-5
                fused_frame_features = total_frame_features / counter
                total_features += fused_frame_features * weight
                
                if do_cache and not cache_hit:
                    feat_2ds = np.stack(feat_2ds, axis=0)
                    cache.add(feat_2ds)
                
            point_ids = torch.unique(frame_visibility_id.nonzero(as_tuple=False)[:, 0])
            mask = torch.zeros(num_points, dtype=torch.bool)
            mask[point_ids] = True
                
            masked_embeddings = total_features[mask]
            
            # print({
            #     "coords": torch.from_numpy(coords), 
            #     "mask": mask,
            #     "masked_embeddings": masked_embeddings
            # })
            
        return {
            "coords": torch.from_numpy(coords), 
            "mask": mask, 
            "masked_embeddings": masked_embeddings
        }
    
    
if __name__ == "__main__":
    
    reader = nuScenesReader("data/sets/v1.0-mini/")

    image_encoder = OpenSegImageEncoder("./exported_model")
    pcd = PointCloudDenseCLIP("data/sets/v1.0-mini/", image_encoder, reader.metadata)
    
    text_encoder = CLIPTextEncoder(device="cpu")
    text_embedding = torch.from_numpy(text_encoder.encode(["Car"])[0])
    
    # del text_encoder
    
    for scene_idx in range(len(reader)):
        reader.set_scene(scene_idx)
        samples_data = reader.get_samples_data()
        output = pcd.get_2d_embeddings(samples_data, neighbor_frame_weights=[0.2, 0.6, 0.2])
        # for k, v in output.items():
        #     print(k, v.shape)
            
        coords = output["coords"]
        mask = output["mask"]
        masked_embeddings = output["masked_embeddings"]
        
        cos_sim = torch.matmul(masked_embeddings, text_embedding)

        full_colors = np.zeros((coords.shape[0], 3), dtype=np.float32)
        valid_mask = cos_sim >= 0
        valid_sim = cos_sim[valid_mask]
        sim_scaled = valid_sim.cpu().numpy()

        colormap = plt.get_cmap('plasma')
        colors_np = colormap(sim_scaled)[:, :3]

        mask_np = mask.cpu().numpy()
        valid_indices_in_masked = torch.nonzero(valid_mask, as_tuple=False).squeeze().cpu().numpy()
        full_indices = np.flatnonzero(mask_np)
        selected_indices = full_indices[valid_indices_in_masked]
        full_colors[selected_indices] = colors_np

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(coords.cpu().numpy())
        pcd.colors = open3d.utility.Vector3dVector(full_colors)
        open3d.io.write_point_cloud("filtered_heatmap_colored.pcd", pcd)


    # scenes_data = reader.get_samples_data()
    
    # first_scene_data = scenes_data[0]
    # print(first_scene_data["CAM_FRONT"])
    
    # filename = first_scene_data["LIDAR_TOP"]["filename"]
    # path = f"{reader.dataroot}/{filename}"
    # points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)

    # coords = np.ascontiguousarray(points[:, :3])
    # print(coords)
    # coords = coords[coords[:, 2] < 0.5]
    # coords = coords[coords[:, 2] > 0]
    
    # x, y = coords[:, 0], coords[:, 1]

    # Define image size and scale
    # img_size = 1000
    # scale = 0.3  # scale factor to convert meters to pixels

    # Shift coordinates to positive values for image coordinates
    # x_img = (x * scale + img_size // 2).astype(np.int32)
    # y_img = (y * scale + img_size // 2).astype(np.int32)

    # Clip to avoid index out of bounds
    # x_img = np.clip(x_img, 0, img_size - 1)
    # y_img = np.clip(y_img, 0, img_size - 1)

    # Create a blank black image
    # img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Draw white pixels at the (x, y) positions
    # img[y_img, x_img] = (255, 255, 255)

    # Save image
    # import cv2
    # cv2.imwrite("point_cloud_xy.png", img)
    
    # reader.get_image_embeddings("scene0", verbose=True)