from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()


@app.get("/")
def root():
    return {"status": "server running"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    contents = await file.read()

    # For now, just confirm image received
    return JSONResponse(
        content={
            "success": True,
            "message": "Image received",
            "filename": file.filename,
            "size_bytes": len(contents)
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
