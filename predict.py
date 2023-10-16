# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, BaseModel, Input, Path, File
import torch
import numpy as np
import cv2
import io
from segment_anything import sam_model_registry, SamPredictor


class Output(BaseModel):
    file: File
    filename: str


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
        source_image: File = Input(description="input image file-like object"),
        model_type: str = Input(
            description="ViT image encoder", 
            choices=["vit_h", "vit_l", "vit_b"], 
            default="vit_h"
        ),
    ) -> Output:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        image_content = source_image.read()
        nparr = np.frombuffer(image_content, np.uint8)
        imagedata = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Check if image was successfully loaded
        if imagedata is None:
            raise ValueError(f"Could not open or read the image")
        
        # Convert the image from BGR to RGB format
        image_rgb = cv2.cvtColor(imagedata, cv2.COLOR_BGR2RGB)

        # output = self.model(processed_image)
        self.predictor.set_image(image_rgb)
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()

        # return postprocess(output)
        # Save the image embedding as a numpy array file
        output_path = "image_embedding.npy"
        np.save(output_path, image_embedding)
        
        # Open the saved file and convert its content to an io.BytesIO instance
        with open(output_path, "rb") as f:
            file_content = io.BytesIO(f.read())

        return Output(filename=output_path, file=file_content)
