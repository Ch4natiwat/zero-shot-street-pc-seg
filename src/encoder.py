import tensorflow.compat.v1 as tf
import tensorflow as tf2
import torch

import matplotlib.cm as cm
import numpy as np
import clip
import cv2


class CLIPTextEncoder:
    
    def __init__(self, device: torch.device):
        
        self.model, _ = clip.load("ViT-L/14@336px", device=device)
        self.device = device
        
        
    def encode(self, texts: list[str]) -> np.ndarray:

        with torch.no_grad():
            
            all_text_embeddings = []
            for text in texts:
                texts = clip.tokenize(text)  # tokenize
                texts = texts.to(self.device)
                text_embeddings = self.model.encode_text(texts)  # embed with text encoder

                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                text_embedding = text_embeddings.mean(dim=0)
                text_embedding /= text_embedding.norm()

                all_text_embeddings.append(text_embedding)

            all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
            all_text_embeddings = all_text_embeddings.to(self.device)
                
        return all_text_embeddings.cpu().numpy().T
    
    
class ImageEncoder:
    
    def encode(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        pass


class OpenSegImageEncoder(ImageEncoder):
    
    def __init__(self, model_dir: str, input_size: tuple[int, int]=(640, 640)):
        
        self.model = tf2.saved_model.load(model_dir, tags=[tf.saved_model.tag_constants.SERVING],)
        self.input_size = input_size
        
        
    def resize_and_pad_image(self, image, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:

        with tf.name_scope("resize_and_pad_image"):
            image_size = tf.cast(tf.shape(image)[0:2], tf.float32)
            scaled_size = target_size

            scale = tf.minimum(scaled_size[0] / image_size[0], scaled_size[1] / image_size[1])
            scaled_size = tf.round(image_size * scale)
            image_scale = scaled_size / image_size

            scaled_image = tf.image.resize_images(
                image, tf.cast(scaled_size, tf.int32), method=tf.image.ResizeMethod.BILINEAR
            )

            output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0, target_size[0], target_size[1])

            image_info = tf.stack([
                image_size,                                  # Original image size
                tf.constant(target_size, dtype=tf.float32),  # New image size
                image_scale,                                 # Resize scales used
                tf.zeros((2,), tf.float32)                   # Offsets (always zeros in our implementation)
            ])
            
            return output_image, image_info
        
        
    def parse_images(self, contents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        def normalize(im):
            im = tf.image.convert_image_dtype(im, tf.float32)
            im -= (128.0 / 255.0)
            im /= (128.0 / 255.0)
            return im

        with tf.Graph().as_default():
            
            raw_images = [tf.io.decode_jpeg(content, channels=3) for content in contents]
            images = [normalize(im) for im in raw_images]
            
            padded_resized_images = []
            images_info = []
            for image in images:
                padded_resized_image, image_info = self.resize_and_pad_image(image, self.input_size)
                padded_resized_images.append(padded_resized_image)
                images_info.append(image_info)

            padded_resized_images = tf.stack(padded_resized_images)
            
            with tf.Session() as sess:
                return sess.run([padded_resized_images, images_info])
        
        
    def preprocess(self, image_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        with tf.gfile.GFile(image_path, "rb") as f:
            model_inputs = np.array([f.read()])
            
        padded_resized_images, images_info = self.parse_images(model_inputs)
        padded_resized_image, image_info = padded_resized_images[0], images_info[0]
        
        crop_size = [
            int(image_info[0, 0] * image_info[2, 0]),
            int(image_info[0, 1] * image_info[2, 1])
        ]
        
        # Revert the original image part of the padded image back to normal
        original_image_part = padded_resized_image[:crop_size[0], :crop_size[1], :]
        original_image_part *= (128.0 / 255.0)
        original_image_part += (128.0 / 255.0)
        padded_resized_image = np.uint8(padded_resized_image * 255)
        
        return padded_resized_image, model_inputs[0]
        
        
    def encode(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        
        padded_resized_image, model_input = self.preprocess(image_path)
        
        # Our CLIP similarity calculation is done outside of OpenSeg,
        # so we give empty text embedding as input
        text_embedding = tf.zeros((1, 1, 768), dtype=tf.float32)

        openseg_outputs = self.model.signatures["serving_default"](
            inp_image_bytes=tf.convert_to_tensor(model_input),
            inp_text_emb=text_embedding
        )
        
        openseg_image_info = openseg_outputs["image_info"]
        crop_size = [
            int(openseg_image_info[0, 0] * openseg_image_info[2, 0]),
            int(openseg_image_info[0, 1] * openseg_image_info[2, 1])
        ]
        embedding = openseg_outputs["ppixel_ave_feat"][0, :crop_size[0], :crop_size[1]]
        resized_image = padded_resized_image[:crop_size[0], :crop_size[1], :]
        
        empty_height = 0
        while tf2.math.count_nonzero(embedding[crop_size[0] - empty_height - 1, :, :]) == 0:
            empty_height += 1
        embedding = embedding[:crop_size[0] - empty_height - 1, :, :]
        resized_image = resized_image[:crop_size[0] - empty_height - 1, :, :]
                
        return resized_image, embedding.cpu().numpy()
        

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    image_encoder = OpenSegImageEncoder("./exported_model")
    text_encoder = CLIPTextEncoder(device=device)

    # Run image encoding
    resized_image, embedding = image_encoder.encode("./data/openseg_examples/police_car.jpg")

    while True:
        prompt = input("Prompt: ")
        if prompt == "EXIT":
            break
        
        # Run text encoding
        prompt_embedding = text_encoder.encode([prompt])[0]
        
        bg_image = cv2.cvtColor(cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        
        similarity = np.tensordot(embedding, prompt_embedding, axes=([2], [0]))
        heatmap = cm.seismic((similarity + 1) / 2)[:, :, :3]
        heatmap = np.uint8(heatmap * 255)
        blended = cv2.addWeighted(bg_image, 0.6, heatmap, 0.4, 0)
        cv2.imwrite("heatmap.jpg", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))