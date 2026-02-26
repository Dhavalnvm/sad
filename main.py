from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, unicodedata, re, logging
from retriever import Retriever
from llm_interface import LLMInterface
from indexer import index_single_file

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str

CHROMA_STORE = "chroma_store"

retriever = Retriever(db_path=CHROMA_STORE)
llm = LLMInterface()

last_answer = ""
MIN_Q = 2
MAX_Q = 2000

def normalize(s: str):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s.lower()).strip()


@app.post("/chat")
async def chat(q: Query, request: Request):
    global last_answer
    txt = q.text.strip()
    if not (MIN_Q <= len(txt) <= MAX_Q):
        return {"response": "Please ask a specific question about the uploaded documents."}

    context, src = retriever.retrieve(txt)

    if context == "NO_RELEVANT":
        return {"response": "I can only answer questions about the uploaded documents."}

    ans = llm.generate_response(txt, context)
    last_answer = ans
    return {"response": ans}


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".csv", ".xlsx", ".txt"]:
        raise HTTPException(400, "Only PDF/CSV/XLSX/TXT allowed.")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file.")

    os.makedirs("data/raw", exist_ok=True)
    safe = file.filename.replace(" ", "_")
    path = f"data/raw/{safe}"

    with open(path, "wb") as f:
        f.write(content)

    index_single_file(path)

    # Reload the store so the retriever picks up the newly indexed vectors
    try:
        retriever.store.load(CHROMA_STORE)
    except Exception as e:
        logger.warning("[upload] Could not reload Chroma store: %s", e)

    return {"message": "File uploaded and indexed.", "filename": safe}


@app.post("/reset")
async def reset(delete_pdfs: bool = False):
    # Wipe and recreate the Chroma store folder
    if os.path.exists(CHROMA_STORE):
        shutil.rmtree(CHROMA_STORE)
    os.makedirs(CHROMA_STORE, exist_ok=True)

    if delete_pdfs and os.path.exists("data/raw"):
        for f in os.listdir("data/raw"):
            if f.lower().endswith((".pdf", ".csv", ".xlsx", ".txt")):
                try:
                    os.remove(f"data/raw/{f}")
                except Exception as e:
                    logger.warning("[reset] Could not delete %s: %s", f, e)

    # Reinitialise the retriever against the now-empty store
    retriever.store.load(CHROMA_STORE)

    return {"message": "Knowledge base reset."}
