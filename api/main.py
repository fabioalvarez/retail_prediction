'''
Minimal example using fastAPI.
'''
from fastapi import FastAPI, UploadFile,File, BackgroundTasks,Request,templating,Form, Cookie, Depends,Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn

# TEST MODULE *--*--*-*-*
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse, Response

# Extra modules
import shutil
import os
from os import getcwd

# Custom modules
from redisqueue.utils import allowed_file
from redisqueue.utils import get_file_hash
from redisqueue.middleware import model_predict
from redisqueue import settings
import shutil


#Instantiate API
app = FastAPI()

#Static file serv
app = FastAPI(title="Service Object Detection")
app.mount("/front",StaticFiles(directory="../api/front"), name="static")

#Jinja2 Template directory
templates = Jinja2Templates(directory="front") #folder templates

# Paths
PATH_FILE = "../data/predictions/"
PATH_ORI = "../data/uploads/"
PATH_DESTORI = "front/assets/temp/"
PATH_DESTINY = "front/assets/temp/imgori/"

# Constrcut endpoints
@app.get("/", response_class=HTMLResponse)
def root():
    html_address = "../api/front/index.html"

    return FileResponse(html_address, status_code=200)


@app.post("/", status_code=201, response_class=HTMLResponse)
async def image(request:Request, response:Response, image: UploadFile = File(...)):

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
        status = model_predict(hashed_name)

        # 4. Update `context` dict with the corresponding values
        path = os.path.join(PATH_FILE ,hashed_name)
        imgpath = os.path.join(PATH_ORI,hashed_name)  

        shutil.copy2(path,PATH_DESTINY + hashed_name)
        shutil.copy2(imgpath,PATH_DESTORI + hashed_name)

        pathpredict = os.path.join(PATH_DESTINY , hashed_name)
        pathori = os.path.join(PATH_DESTORI , hashed_name)
        
        return templates.TemplateResponse("index.html",{"request":request,"imgpredict":pathpredict,"imgori":pathori})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)