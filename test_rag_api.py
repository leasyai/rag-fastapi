from fastapi.testclient import TestClient
#from main import get_top_k_embeddings, VectorStoreQuery, VectorStore, Vector_Retriever
import pytest
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters, MetadataFilter
from unittest.mock import patch, AsyncMock, MagicMock
import rag_api
from rag_api import get_top_k_embeddings, VectorStoreQuery, VectorStore, Vector_Retriever

client = TestClient(rag_api.app)


#client = TestClient(app)


def test_top_k_embeddings():

    query_embedding = [1.0, 0.0]

    doc_embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ]

    doc_ids = ["doc1", "doc2", "doc3"]

    similarities, ids = get_top_k_embeddings(
        query_embedding,
        doc_embeddings,
        doc_ids,
        similarity_top_k=2
    )

    assert ids == ["doc1", "doc2"]




def test_vector_store_query():

    # Create real vector store
    vector_store = VectorStore()

    # Create nodes with simple embeddings
    node1 = TextNode(text="Document about cats", metadata={"category": "animals"})
    node1.embedding = [1.0, 0.0]

    node2 = TextNode(text="Document about trees", metadata={"category": "plants"})
    node2.embedding = [0.0, 1.0]

    node3 = TextNode(text="Unrelated document", metadata={"category": "items"})
    node3.embedding = [-1.0, 0.0]

    # Add nodes to vector store
    vector_store.add([
        node1,
        node2,
        node3
    ])
    
    filters = MetadataFilters(filters=[
        MetadataFilter(
            key="category",
            value="animals"
        )
    ]
)

           
    

    # Query embedding is closest to node1
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=1, 
        filters= filters   
    )

    result = vector_store.query(query)

    assert len(result.nodes) == 1 
    

    assert result.nodes[0].node_id == node1.node_id


class FakeEmbeddingModel:
    def get_query_embedding(self, query: str):
        return [1.0, 0.0]

        
def test_vector_retriever():

    # 1. Create vector store
    vector_store = VectorStore()

    # 2. Create nodes with known embeddings
    node1 = TextNode(text="Cats are animals")
    node1.embedding = [1.0, 0.0]

    node2 = TextNode(text="Computers are machines")
    node2.embedding = [0.0, 1.0]

    node3 = TextNode(text="Opposite direction")
    node3.embedding = [-1.0, 0.0]

    vector_store.add([
        node1,
        node2,
        node3
    ])

    # 3. Fake embedding model
    embed_model = FakeEmbeddingModel()

    # 4. Create your real retriever
    retriever = Vector_Retriever(
        vector_store=vector_store,
        embed_model=embed_model,
        similarity_top_k=2
    )

    # 5. Retrieve
    results = retriever.retrieve(
        "Which document is most similar?"
    )

    # 6. Check results
    assert len(results) == 2

    assert results[0].node.node_id == node1.node_id
    assert results[1].node.node_id == node2.node_id

    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query):
        return self.results
        
def test_get_answer():
    
    node = TextNode(
        text="Llama is a family of large language models."
    )

    fake_retrieval = [
        NodeWithScore(
            node=node,
            score=0.9
        )
    ]

    fake_response = MagicMock()

    fake_response.json.return_value = {
        "answer": "Llama is a transformer-based language model."
    }
    
    fake_response.raise_for_status.return_value = None
    fake_retriever = FakeRetriever(fake_retrieval)
    with TestClient(rag_api.app) as client:

        # IMPORTANT:
        # set retriever on EXACTLY the app used inside rag_api.py
        rag_api.app.state.retriever = fake_retriever
        with patch(
        "rag_api.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=fake_response
        ):

            response = client.post(
                "/get_answer",
                json={
                    "query": "What is Llama?",
                    "top_k": 3
                }
            )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "Llama is a transformer-based language model."
    )

    assert len(data["sources"]) == 1

    assert data["sources"][0]["similarity"] == pytest.approx(0.9)