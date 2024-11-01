import cv2
import torch
import numpy as np
from numpy.typing import NDArray
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import requests
from typing import Any


def sam_predictor():
    checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    return predictor
    

# Segment func
def segmented(input_json: Any, predictor: SAM2ImagePredictor) -> None:
    if input_json.get('url') is None:
        return []
    else:
        image_url = input_json.get('url')
        response = requests.get(image_url)
        image_np = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_point = np.array([[input_json.get('x'), input_json.get('y')]])
        input_label = np.ones((input_point.shape[0],), dtype=int)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image_rgb)
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        first_mask = masks[0]
        first_mask = first_mask.astype(np.uint8)
        output = cv2.bitwise_and(image_rgb, image_rgb, mask=first_mask)

        return output
