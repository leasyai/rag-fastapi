
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.vector_stores import (VectorStoreQuery,VectorStoreQueryResult)
from typing import List, Any, Optional, Dict
from llama_index.core.schema import TextNode, BaseNode, Field
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.query_engine import RetrieverQueryEngine
import os
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from typing import Tuple, cast, Any, List, Optional
import numpy as np
from contextlib import asynccontextmanager
import tempfile




import torch
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel






async def model_load(app: FastAPI):
      
    model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

    
    app.state.llm = LlamaCPP( model_url=model_url, model_path=None, temperature=0.1,max_new_tokens=256,
               context_window=2024,
               # kwargs to pass to __call__()
               generate_kwargs={},
               # kwargs to pass to __init__()
               # to use GPU
               model_kwargs={"n_gpu_layers":-1},# run all layers on gpu
               verbose=True,
              )
    
    

    


    
    

    yield
    
    
    app.state.llm = None
    


app = FastAPI(lifespan=model_load)

#---------------------------------------------------
# Query Request Model
# ---------------------------------------------------

class QueryRequest(BaseModel):

    prompt: str
    



#-------------------------------------------------------------------
#Get the answer answer..
#-------------------------------------------------------------------

@app.post("/generate")
async def get_answer(request: QueryRequest):

    response = app.state.llm.complete(
        request.prompt)
    return {
        "answer": str(response)
    }