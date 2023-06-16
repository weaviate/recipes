from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
import weaviate

# connect to your weaviate instance
client = weaviate.Client("http://localhost:8080")

# load the blogs in using the reader
blogs = SimpleDirectoryReader('./data').load_data()

# chunk up the blog posts into nodes 
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(blogs)


from llama_index.vector_stores import WeaviateVectorStore
from llama_index import VectorStoreIndex, StorageContext
from llama_index.storage.storage_context import StorageContext


# construct vector store
vector_store = WeaviateVectorStore(weaviate_client = client, index_name="BlogPost", text_key="content")

# setting up the storage for the embeddings
storage_context = StorageContext.from_defaults(vector_store = vector_store)

# set up the index
index = VectorStoreIndex.from_documents(nodes, storage_context = storage_context)

# and now query 🚀
query_engine = index.as_query_engine()
response = query_engine.query("What is the intersection between LLMs and search?")
print(response)