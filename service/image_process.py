import cv2
import base64
import numpy as np


async def read_image(file):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return None

    image_resized = cv2.resize(image, (640, 640))
    return image_resized


def image2bytes(image):
    _, img_encoded = cv2.imencode(".jpg", image)
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
    return img_base64



def image_to_img_src(image) -> str:
    return f"data:image/png;base64,{image}"
