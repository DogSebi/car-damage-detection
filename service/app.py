import cv2
import base64
import uvicorn
import traceback
import asyncio

from fastapi import FastAPI, UploadFile, File
from fastapi import Request, Response, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

from service.models.yolo_model import predict_parts
from service.models.segformer_model import predict_damage
from service.find_damage import analyze_damage
from service.image_process import read_image, image_to_img_src, image2bytes

app = FastAPI()
templates = Jinja2Templates(directory="service/templates")


@app.get("/", response_class=HTMLResponse)
def root(request: Request) -> Response:
    return templates.TemplateResponse(request, "index.html")


@app.post("/", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)):
    ctx = {}
    try:
        image = await read_image(file)
        damage_task = asyncio.to_thread(predict_damage, image)
        parts_task = asyncio.to_thread(predict_parts, image)
        damage_mask = await damage_task
        parts_masks, part_names_for_masks = await parts_task

        result_img, report = analyze_damage(
            parts_masks=parts_masks,
            damage_mask=damage_mask,
            original_image=image,
            part_names=part_names_for_masks
        )

        predicted_image = image2bytes(result_img)
        original_image = image2bytes(image)

        ctx.update(
            predicted_image=image_to_img_src(predicted_image),
            original_image=image_to_img_src(original_image),
            report=report
        )

        return templates.TemplateResponse(request, "index.html", ctx)

    except Exception as err:
        ctx.update(error=str(err), error_name=type(err).__name__)
        return templates.TemplateResponse(request, "index.html", ctx)


@app.post("/api")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image = await read_image(file)
        damage_task = asyncio.to_thread(predict_damage, image)
        parts_task = asyncio.to_thread(predict_parts, image)
        damage_mask = await damage_task
        parts_masks, part_names_for_masks = await parts_task

        result_img, report = analyze_damage(
            parts_masks=parts_masks,
            damage_mask=damage_mask,
            original_image=image,
            part_names=part_names_for_masks
        )

        if not report:
            report.append("Повреждений не выявлено")

        img_base64 = image2bytes(result_img)

        return JSONResponse(content={
            "report": report,
            "image_base64": img_base64
        })

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
