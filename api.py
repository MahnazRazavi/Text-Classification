from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import tensorflow as tf
from test import TextClassificatin
from options import TextOptions

app = FastAPI()

text_classifier = None
class_id_map = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_classifier
    global class_id_map
    text_classifier = TextClassificatin()
    class_id_map = {"0": "Neutral", "1": "Xenophobic", "2": "Racist"}
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Text Classification API",
    description="""
    A FastAPI application for text classification.

    This API allows you to classify text data using a pre-trained model.
    The model is loaded once at startup and is used for all subsequent inference requests.

    ## Endpoints

    - **/classify**: POST endpoint to classify a given text input.
    """,
    version="0.1.0",
)


class TextRequest(BaseModel):
    text: str


# Default route to show the API description
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Text Classification API</title>
        </head>
        <body>
            <h1>Text Classification API</h1>
            <p>A FastAPI application for text classification.</p>
            <p>This API allows you to classify text data using a pre-trained model.
            The model is loaded once at startup and is used for all subsequent inference requests.</p>
            <h2>Endpoints</h2>
            <ul>
                <li><b>/classify</b>: POST endpoint to classify a given text input.</li>
            </ul>
        </body>
    </html>
    """


# /classify route for inference
@app.post("/classify")
async def classify_text(request: TextRequest):
    if not text_classifier:
        raise HTTPException(status_code=500, detail="Model not loaded")

    result = text_classifier.text_classification(request.text)
    result = {
        "text": request.text,
        "class_id": str(result.item()),
        "class_name": class_id_map[str(result.item())],
    }

    return result
