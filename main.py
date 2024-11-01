from fastapi import FastAPI, Request
from utils import segmented, sam_predictor
from infer import run_triplane_gaussian_splatting
from PIL import Image
import numpy as np
import io
from fastapi.responses import Response

cached_sam2_model = sam_predictor()

app = FastAPI()

@app.post("/process-image/")
async def get_item(request: Request):
    input_json = await request.json()  # Parse the JSON body
    output = segmented(input_json, cached_sam2_model)

    # Convert output to PNG
    image_pil =  Image.fromarray((output).astype(np.uint8))
    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Return raw bytes with appropriate content type
    return Response(content=buffer.getvalue(), media_type="image/png")

@app.post("/triplane/")
async def get_item(request: Request):
    input_json = await request.json()  # Parse the JSON body
    output = run_triplane_gaussian_splatting(input_json.get('url'), cam_dist=input_json.get('cam_dist'))
    return {"plybytes": output}

