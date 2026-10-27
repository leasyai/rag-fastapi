import pytest
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters, MetadataFilter
from unittest.mock import patch, AsyncMock, MagicMock
import main
from main import store_nodes


        
def test_store_nodes(monkeypatch):
    # 2. Create nodes with known embeddings
    
    monkeypatch.setenv("DB_PASSWORD", "dummy_test_password")
    
    node1 = TextNode(text="Cats are animals")
    node1.embedding = [1.0, 0.0]

    node2 = TextNode(text="Computers are machines")
    node2.embedding = [0.0, 1.0]

    node3 = TextNode(text="Opposite direction")
    node3.embedding = [-1.0, 0.0]

    nodes = [node1, node2, node3]

    fake_connection = MagicMock()
    fake_cursor = MagicMock()
    fake_connection.cursor.return_value.__enter__.return_value= fake_cursor
    with patch("main.psycopg.connect",
            
            return_value=fake_connection
            ):
            
            store_nodes(nodes)
          
    fake_connection.cursor.return_value.__enter__.return_value= fake_cursor
             
    assert fake_cursor.execute.call_count ==3         
    print(fake_cursor.execute.call_args_list)