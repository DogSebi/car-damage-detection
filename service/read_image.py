import cv2
import numpy as np


async def read_image(file):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return None

    image_resized = cv2.resize(image, (640, 640))
    return image_resized

def image_to_img_src(image) -> str:
    return f"data:image/png;base64,{image}"
