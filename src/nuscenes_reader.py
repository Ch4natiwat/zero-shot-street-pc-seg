from nuscenes.nuscenes import NuScenes
import cv2
import os


class nuScenesReader:
    
    def __init__(self, dataroot: str):
        self.dataroot = dataroot
        self.nusc = NuScenes(dataroot=dataroot, verbose=False)
        self.scene = self.nusc.scene[0]
        self.cam_keys = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
        self.lidar_key = "LIDAR_TOP"
        self.sensor_keys = self.cam_keys + [self.lidar_key]
        self.image_dim = self.get_image_dim()
        self.metadata = self.get_metadata()
        
        
    def __len__(self):
        return len(self.nusc.scene)
    
    
    def get_image_dim(self):
        sample = self.nusc.get("sample", self.scene["first_sample_token"])
        cam_data = self.nusc.get("sample_data", sample["data"][self.cam_keys[0]])
        sample_image_path = os.path.join(self.dataroot, cam_data["filename"])
        sample_image_dim = cv2.imread(sample_image_path).shape
        return (sample_image_dim[0], sample_image_dim[1])
    
    
    def get_metadata(self):
        return {
            "image_dim": self.image_dim,
            "cam_keys": self.cam_keys,
            "lidar_key": self.lidar_key
        }
    
    
    def set_scene(self, idx: int):
        self.scene = self.nusc.scene[idx]
        
        
    def get_samples_data(self):
        samples_data = []
        sample_token = self.scene["first_sample_token"]
        
        while sample_token:
            sample = self.nusc.get("sample", sample_token)
            sample_sensor_readings = {
                "timestamp": sample["timestamp"]
            }
            
            all_sample_data = sample["data"]
            for sensor in self.sensor_keys:
                sample_data = self.nusc.get("sample_data", all_sample_data[sensor])
                ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])
                ego_pose = {
                    "translation": ego_pose["translation"], 
                    "rotation": ego_pose["rotation"]
                }
                calibrated_sensor = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
                calibrated_sensor = {
                    "translation": calibrated_sensor["translation"], 
                    "rotation": calibrated_sensor["rotation"],
                    "camera_intrinsic": calibrated_sensor["camera_intrinsic"]
                }
                filename = sample_data["filename"]
                sensor_modality = sample_data["sensor_modality"]
                sample_sensor_readings[sensor] = {
                    "sensor_modality": sensor_modality,
                    "ego_pose": ego_pose,
                    "calibrated_sensor": calibrated_sensor,
                    "filename": filename
                }
                
            samples_data.append(sample_sensor_readings)
            sample_token = sample["next"]
            
        return samples_data