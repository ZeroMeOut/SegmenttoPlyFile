import requests
import json
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define the URL
url = 'http://localhost:8000/process-image/'

# Define the payload (data to be sent in the body of the POST request)
payload = {
    "url": "https://fastly.picsum.photos/id/237/200/300.jpg?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U",
    "x": 100,
    "y": 60
}

# Define the headers
headers = {
    'Content-Type': 'application/json'
}

# Send the POST request with the JSON payload
sam_response = requests.post(url, headers=headers, data=json.dumps(payload))
org_image = Image.open(BytesIO(requests.get(payload['url']).content))


# Check the response
print("Status Code:", sam_response.status_code)
#print("Response Body:", response.json())
mask = Image.open(BytesIO(sam_response.content))
org_image.show()
mask.show()

