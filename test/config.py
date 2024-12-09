"""
MOST of the modules will REQUIRE config to be loaded, and use objects initialized in this module.

This module describes and builds memory units (<=>ragraph graphs), and the elementary operations.
"""

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

import random
from difflib import SequenceMatcher
from src import ChromaGraph
from raphlib import LLMFunction, setup_env
from langchain_openai import AzureChatOpenAI

# ================================================================= CONSTANTS

PERSIST_DIRECTORY = "./data/chroma"

# ================================================================= MERGER

LLM = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",
    temperature=0.2,
    max_tokens=500,
)

llm_merger = LLMFunction(
    LLM,
    """
    Produce a new chunk that contains the information from the previous two chunks.
    Make the new chunk synthetic, minimal.
    Don't write anything but the new chunk in your next response.

    CHUNK1:
    {chunk}

    CHUNK2:
    {other}
    """,
    chunk=str
)

def merge_function(chunk: str, other: str) -> str:
    
    if SequenceMatcher(None, chunk, other).ratio() >= 0.8:
        chosen = random.choice((chunk, other))
        if chunk != other:
            print("> 80% SIMILAR, RANDOMLY SELECTED", chosen, "FROM", chunk, "|", other)
        return chosen
    
    else:
        result = llm_merger.invoke({"chunk": chunk, "other": other})
        print("< 80% SIMILAR, MERGED AS", result.chunk, "MERGED", chunk, "|", other)
        return result.chunk

# ================================================================= DEFINIG DATABASE

client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_function = model.encode

# ================================================================= COLLECTIONS

test = ChromaGraph(
    client, 
    "test", 
    embedding_function, 
    merge_function, 
    merge_function,
    content_equality_threshold = 0.5,
    accessor_equality_threshold = 0.5,
)

# ================================================================= OPERATIONS



# ================================================================= TEST

# Create a node from a chunk

a1 = test.Accessor("Cats")

a1.link(test.Content("Cats like fish"))
a1.link(test.Content("Cats don't like swimming."))

a2 = test.Accessor("Fish")

a2.link(test.Content("Fish are eaten by cats."))
a2.link(test.Content("Fish live in water."))

c1 = test.Content("Cats like to eat fish.")

a1.link(c1)

c2 = test.Content("Cats like eating fish.")

a2.link(c2)

# Commit the node

print(a2.neighbors)

test.commit()

# ================================================================= DISPLAY

from graph_display import display_plotly

display_plotly(test)
