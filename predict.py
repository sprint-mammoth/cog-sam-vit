# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, File
import torch
import numpy as np
import cv2
import io
import tempfile
from segment_anything import sam_model_registry, SamPredictor


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device='cuda')
        self.predictor = SamPredictor(sam)

    def predict(
        self,
        source_image: Path = Input(description="input image file handler"),
    ) -> Path:
        """Run a single prediction on the model"""
        try:
            # processed_input = preprocess(image)
            imagedata = cv2.imread(str(source_image))

            # Check if image was successfully loaded
            if imagedata is None:
                raise ValueError(f"Could not open or read the image")
            
            # Convert the image from BGR to RGB format
            image_rgb = cv2.cvtColor(imagedata, cv2.COLOR_BGR2RGB)

            # output = self.model(processed_image)
            self.predictor.set_image(image_rgb)
            image_embedding = self.predictor.get_image_embedding().cpu().numpy()

            print(image_embedding.shape)
            print("successfully make image embedding\n")

            # return postprocess(output)
            ''' # At the start of your predict function, use a NamedTemporaryFile:
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as temp_file:
                if temp_file is None:
                    raise ValueError(f"Could not create temporary file")
                # Save your numpy array to this file
                np.save(temp_file, image_embedding)
                # Ensure file pointer is at the beginning
                temp_file.seek(0)
                # Return the file handle
                return File(temp_file)'''
        
            # Save the image embedding to a temporary numpy array file
            # This file will automatically be deleted by Cog after it has been returned.
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as temp_file:
                np.save(temp_file.name, image_embedding)
                temp_path = temp_file.name
                print(f"embedding file is saved to temp_path: {temp_path}\n")

            return Path(temp_path)
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")
