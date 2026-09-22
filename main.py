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
import json
import psycopg



import torch
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel


# ---------------------------------------------------
# Global variables
# ---------------------------------------------------

nodes=[]
query=''
llm = None


# ---------------------------------------------------
# PDF Processing
# ---------------------------------------------------

def process_pdf(path):
    loader = PyMuPDFReader()
    documents = loader.load(path)
    parser = SentenceSplitter(chunk_size=1000)# 1000 tokens
    
    nodes = parser.get_nodes_from_documents(documents)
    return nodes




# ---------------------------------------------------
# Custom Vector Store
# ---------------------------------------------------

class  VectorStore(BasePydanticVectorStore):
    
    stores_text: bool = True
    node_dict: Dict[str, BaseNode] = Field(default_factory=dict)
    @property
    def client(self):
        return None

    def get(self, text_id: str) -> List[float]:
        # Get embedding.
        return self.node_dict[text_id]

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        # Add nodes to index.
        for node in nodes:
            self.node_dict[node.node_id] = node

    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        
        # Delete nodes using with node_id.

        del self.node_dict[node_id]

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        # Get nodes for response.
        #  filter by metadata 
        nodes = self.node_dict.values()
        if query.filters is not None:
            nodes = filter_nodes(nodes, query.filters)
        if len(nodes) == 0:# no returns
            result_nodes = []
            similarities = []
            node_ids = []
        else:
            # if filteration returns any results, apply semantic search
            similarities, node_ids = dense_search(query, nodes)
            result_nodes = [self.node_dict[node_id] for node_id in node_ids]
        return VectorStoreQueryResult(
            nodes=result_nodes, similarities=similarities, ids=node_ids
        )   

    def persist(self, persist_path, fs=None) -> None:
        # Persist the SimpleVectorStore to a directory.
        # we are not implementing this for now.

        pass


#----------------------------------------------------
# database store
#----------------------------------------------------



def store_nodes(nodes):

    conn = psycopg.connect(
        host="localhost",
        port=5432,
        dbname="rag_db",
        user="layal",
        password="your_password"
    )

    with conn.cursor() as cur:

        for node in nodes:

            cur.execute(
                """
                INSERT INTO nodes (id, text, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                (
                    node.node_id,
                    node.get_content(),
                    node.embedding,
                    json.dumps(node.metadata)
                )
            )

    conn.commit()
    conn.close()


#-----------------------------------------------------
#query
#------------------------------------------------------
def query(
    query: VectorStoreQuery,
    **kwargs: Any,
) -> VectorStoreQueryResult:

    conn = psycopg.connect(
        host="localhost",
        port=5432,
        dbname="rag_db",
        user="layal"
    )

    try:
        with conn.cursor() as cur:

            sql = """
                SELECT
                    id,
                    text,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM nodes
            """

            params = [query.query_embedding]

            # ---------------------------------
            # Metadata filtering
            # ---------------------------------
            if query.filters is not None:

                conditions = []

                for f in query.filters.filters:
                    conditions.append(
                        "metadata ->> %s = %s"
                    )

                    params.append(f.key)
                    params.append(str(f.value))

                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)

            # ---------------------------------
            # Similarity ranking
            # ---------------------------------
            sql += """
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """

            params.append(query.query_embedding)
            params.append(query.similarity_top_k)

            cur.execute(sql, params)

            rows = cur.fetchall()

    finally:
        conn.close()

    result_nodes = []
    similarities = []
    node_ids = []

    for row in rows:

        node_id = row[0]
        text = row[1]
        metadata = row[2]
        similarity = row[3]

        node = TextNode(
            id_=node_id,
            text=text,
            metadata=metadata or {}
        )

        result_nodes.append(node)
        similarities.append(float(similarity))
        node_ids.append(node_id)

    return VectorStoreQueryResult(
        nodes=result_nodes,
        similarities=similarities,
        ids=node_ids
    )
# ---------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------
def get_top_k_embeddings(
    query_embedding: List[float],
    doc_embeddings: List[List[float]],
    doc_ids: List[str],
    similarity_top_k: int = 5,
) -> Tuple[List[float], List]:
    
    # dimensions: D
    qembed_np = np.array(query_embedding)
    # dimensions: N x D
    dembed_np = np.array(doc_embeddings)
    # dimensions: N
    dproduct_arr = np.dot(dembed_np, qembed_np)
    # dimensions: N
    norm_arr = np.linalg.norm(qembed_np) * np.linalg.norm(
        dembed_np, axis=1, keepdims=False
    )
    # dimensions: N
    cos_sim_arr = dproduct_arr / norm_arr

    # now we have the N cosine similarities for each document, sort by top k cosine similarity, and return ids
    tups = [(cos_sim_arr[i], doc_ids[i]) for i in range(len(doc_ids))]
    sorted_tups = sorted(tups, key=lambda t: t[0], reverse=True)

    sorted_tups = sorted_tups[:similarity_top_k]

    result_similarities = [s for s, _ in sorted_tups]
    result_ids = [n for _, n in sorted_tups]
    return result_similarities, result_ids



# # ---------------------------------------------------
# Metadata Filtering: filteration based on metadata of both query and nodes
# ---------------------------------------------------

def filter_nodes(nodes: List[BaseNode], filters: MetadataFilters):
    filtered_nodes = []
    for node in nodes:
        matches = True
        for f in filters.filters:
            if f.key not in node.metadata:
                matches = False
                continue
            if f.value != node.metadata[f.key]:
                matches = False
                continue
        if matches:
            filtered_nodes.append(node)
    return filtered_nodes   


# ---------------------------------------------------
# Dense Search
# ---------------------------------------------------

def dense_search(query: VectorStoreQuery, nodes: List[BaseNode]):
    """Dense search."""
    query_embedding = cast(List[float], query.query_embedding)
    doc_embeddings = [n.embedding for n in nodes]
    doc_ids = [n.node_id for n in nodes]
    return get_top_k_embeddings(
        query_embedding,
        doc_embeddings,
        doc_ids,
        similarity_top_k=query.similarity_top_k,
    )    



# ---------------------------------------------------
# FastAPI Lifespan
# ---------------------------------------------------

async def model_load(app: FastAPI):
      
    model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

    app.state.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en") 
    app.state.llm = LlamaCPP( model_url=model_url, model_path=None, temperature=0.1,max_new_tokens=256,
               context_window=2500,
               # kwargs to pass to __call__()
               generate_kwargs={},
               # kwargs to pass to __init__()
               # to use GPU
               model_kwargs={"n_gpu_layers":-1},# run all layers on gpu
               verbose=True,
              )
    
    
    


    app.state.vector_store = VectorStore()
    app.state.retriever= Vector_Retriever(vector_store=app.state.vector_store,
        embed_model=app.state.embed_model,
        similarity_top_k=3
    )
    

    yield
    
    app.state.embed_model = None
    app.state.llm = None
    app.state.vector_store = None
    app.state.retriever = None


app = FastAPI(lifespan=model_load)

#------------------------------------------------------------------------
# build a retriever
#------------------------------------------------------------------------


class Vector_Retriever(BaseRetriever):
    """Retriever over in_memory vector store."""

    def __init__(
        self,
        #vector_store: Any,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 3,
    ) -> None:
        """Init params."""
        #self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = "default"
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        ) # construct vector store query
        query_result = query(vector_store_query)# returns a VectorStoreQueryResult based on similarity scores
        
        # Parse Result into a Set of Nodes
        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores
# ---------------------------------------------------
# Upload PDF
# ---------------------------------------------------
@app.post("/pdf")
async def upload_file(file:UploadFile):
    global nodes
    content = await file.read()
    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as temp_file:

        temp_file.write(content)
        temp_path = temp_file.name
    nodes = process_pdf(temp_path)
    # Generate embeddings
    # only AFTER nodes exist
    for node in nodes:

        node_embedding = (
            app.state.embed_model
            .get_text_embedding(
                node.get_content(
                    metadata_mode="all"
                )
            )
        )

        node.embedding = (
            node_embedding
        )

    # Add embedded nodes
    # to vector store
    #app.state.vector_store.add(
       # nodes
   # )
    store_nodes(nodes)
    return {
        "number_of_nodes":
        len(nodes)
    }
    




#---------------------------------------------------
# Query Request Model
# ---------------------------------------------------

class QueryRequest(BaseModel):

    query: str
    top_k: int = 3



#-------------------------------------------------------------------
#Get the answer answer..
#-------------------------------------------------------------------

@app.post("/get_answer")
async def get_answer(request: QueryRequest):
    
    query_engine = RetrieverQueryEngine.from_args(
        app.state.retriever,
        llm=app.state.llm
    )

    response = query_engine.query(
        request.query
    )
    #response= app.state.llm.complete(request.query)
    return {
        "query": request.query,
        "answer": str(response),
        "source": response.source_nodes[0].get_content()
        }
#---------------------------------------------------
# automated test: Health
# ---------------------------------------------------    
@app.get("/health")
async def health():
    return {"status": "ok"}