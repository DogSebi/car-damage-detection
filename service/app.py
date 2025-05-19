import cv2
import base64
import uvicorn
import traceback

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from service.models.yolo_model import predict_parts
from service.models.segformer_model import predict_damage
from service.find_damage import analyze_damage
from service.read_image import read_image

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Car damage detection service is running."}


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):

    try:
        image = await read_image(file)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        damage_mask = predict_damage(image)
        parts_masks, part_names_for_masks = predict_parts(image)

        result_img, report = analyze_damage(
            parts_masks=parts_masks,
            damage_mask=damage_mask,
            original_image=image,
            part_names=part_names_for_masks
        )

        if not report['damaged_parts']:
            report["damaged_parts"] = 'Повреждений не выявлено'

        _, img_encoded = cv2.imencode(".jpg", result_img)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

        return JSONResponse(content={
            "report": report,
            "image_base64": img_base64
        })

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
