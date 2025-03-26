from fastapi import FastAPI, UploadFile, File
import uvicorn
import shutil
import os
from app.trainer import train_lora, log_path
from app.ui import launch_ui

app = FastAPI()
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "uploaded", "filename": file.filename}

@app.post("/train")
def train_endpoint():
    train_lora()
    return {"status": "training started"}

@app.get("/logs")
def get_logs():
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return {"logs": f.read()}
    return {"logs": "No logs yet"}

@app.on_event("startup")
def startup_event():
    launch_ui()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)