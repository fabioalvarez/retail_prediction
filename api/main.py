'''
Minimal example using fastAPI.
'''
from fastapi import FastAPI, UploadFile,File, BackgroundTasks,requests,templating
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Extra modules
import shutil
import os
from os import getcwd

# Custom modules
from utils import allowed_file
from utils import get_file_hash
from middleware import model_predict
import settings

# Instantiate API

app = FastAPI()

app = FastAPI(title="Service Object Detection")
app.mount("/front",StaticFiles(directory="../api/front"), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    html_address = "../api/front/index.html"

    return FileResponse(html_address, status_code=200)

@app.post("/", status_code=201)
async def image(image: UploadFile = File(...)):

    # Get image name
    file_name = image.filename

    # No file received, show basic UI
    if not image:
        return {"message": "There is no image"}

    # File received but no filename is provided, show basic UI
    elif file_name == "":
        return {"message": "No image selected for uploading"}


    # File received and it's an image, we must show it and get predictions
    elif image and allowed_file(file_name):
        # In order to correctly display the image in the UI and get model
        # predictions we implement the following:

        # 1. Get an unique file name using utils.get_file_hash() function
        # Create full path to save image
        hashed_name = get_file_hash(image.file, file_name)
        save_path = os.path.join(settings.UPLOAD_FOLDER, hashed_name)

         # 2. Store the image to disk using the new name.
        with open(f"{save_path}", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # 3. Send the file to be processed by the `model` service            
        prediction, score = model_predict(hashed_name)

        # 4. Update `context` dict with the corresponding values
        context = {
            "prediction": prediction,
            "score": str(score),
            "filename": hashed_name}
    
    return context


@app.post("/hash")
async def image(image: UploadFile = File(...)):

    # Get image name
    file_name = image.filename
    hashed_name = get_file_hash(image.file, file_name)
    return {"hash_name": hashed_name}


    ###################


# from fastapi import FastAPI, UploadFile,File, BackgroundTasks,requests,templating
# from fastapi.responses import HTMLResponse, FileResponse
# from fastapi.staticfiles import StaticFiles
# from os import getcwd

# import uuid
# db = []
# PATH_FILE = "../api/feedback/" #folder file predict


# app = FastAPI(title="Service Object Detection")
# app.mount("/templates",StaticFiles(directory="../api/templates"), name="static")


# @app.get("/", response_class=HTMLResponse)
# def root():

#     html_address = "../api/templates/index.html"
#     return FileResponse(html_address, status_code=200)

# """
# Function used in the frontend so it can upload and show an image.

# """

# @app.post("/")
# async def uploadfile(file:UploadFile= File(...)):
#     #file.filename = f"{uuid.uuid4()}.jpg"
#     file_name = file.filename
#     contents = await file.read()
#     db.append(contents)
#     path = PATH_FILE + file_name #folder file and filename(hash)
#    # html_result = "../api/templates/result.html"

#     return FileResponse(path=path)

