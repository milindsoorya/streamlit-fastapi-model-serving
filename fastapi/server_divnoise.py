import io

from models.divnoising import divnoise
from starlette.responses import Response

from fastapi import FastAPI, File
import uvicorn

model = divnoise.get_model()

app = FastAPI(
    title="Image denoising using Deeplearning",
    description="""Model used is DivNoising trained on mouse nuclei""",
    version="0.1.0",
)


@app.post("/denoise")
def get_denoise_image(file: bytes = File(...)):
    """Denoise the image file"""
    denoised_image = divnoise.denoise_image(model, file)
    bytes_io = io.BytesIO()
    denoised_image.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)