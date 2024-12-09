"""
Calls specific modules to create a database with a particular number of collections.
"""
import chromadb
from typing import List
from raphlib import LapTimer, LLMFunction, setup_env
from memory import BaseMemory
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

Chunk = str

DTIME = LapTimer()
print("INTIALIZING DATABASE")

PERSIST_DIRECTORY = "./chroma"

client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'  # Ensure this matches the model used below
        )

DTIME.lap("\nINITIALIZING LLM\n\n")

from langchain_openai import AzureChatOpenAI

setup_env()

LLM = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",
    temperature=0.2,
    max_tokens=500,
)

def EXTRACT_CONTEXT(input: str) -> Chunk:
        """
        Create a context chunk from the input string.
        """
        result = LLMFunction(LLM, 
            "Fill the topic field with a short sentence that summarize the main thing the following conversation seems to be about :\n{input}", 
            topic=str,
            
        ).invoke({"input": input})
        return result.topic
    
def EXTRACT_CONTENT(input: str) -> List[Chunk]:
    """
    Extract a list of content chunks.
    """
    result = LLMFunction(LLM,
        """Extract information from this paragraph as atomic and independant chunks of information, 
        to be stored in a knowledge base. 
        Separate the information that is about the conversation itself (wants and actions of participants)
        from factual information about the world if there is any, as "conversation_knowledge" and "factual_knowledge" respectively.
        PARAGRAPH : {input}""", 
        conversation_knowledge=["Raphael asked about potatoes", ...],
        factual_knowledge=["Raphael likes potatoes",...]
        ).invoke({"input": input})
    return result.factual_knowledge

instructions = BaseMemory(client, model, "instructions", embedding_function, use_metadata=True)

DTIME.lap("EXTRACTING CONTENT")

input = """
User: Hello I want to know more about cats.
Ai: Sure, cats are fluffly.
User: Are you sure about that ?
"""

chunk = EXTRACT_CONTEXT(input)
print(chunk, "\n")

DTIME.lap("EXTRACTING CONTENT")

contents = EXTRACT_CONTENT(input)
print(contents, "\n")

DTIME.lap("ADDING ACCESSOR")

cats_context = instructions.ADD_ACCESSOR(chunk)

DTIME.lap("ADDING CONTENT")

for content in contents:
    instructions.ADD_CONTENT(cats_context, content)

DTIME.lap("ADDING ACCESSOR AND CONTENT PAIRS")

building_context = instructions.ADD_ACCESSOR("A conversation about building homes.")
cold_context = instructions.ADD_ACCESSOR("A conversation about feeling cold.")

for content in ["Laura's dog likes to sleep on her feet.", "Raphael does not like winter.", "Cats are staying inside during winter."]:
    instructions.ADD_CONTENT(cold_context, content)

for content in ["Laura likes it when her dog sleep on her feet.", "Laura does not like building homes.", "Cats need to stay inside the buildings during winter.", "Cats are fluffy"]:
    instructions.ADD_CONTENT(building_context, content)

table_content = instructions.ADD_ACCESSOR("They are talking about ikea furnitures")
instructions.ADD_CONTENT(table_content, "Raphael bought his chair from ikea")
instructions.ADD_CONTENT(table_content, "Tables are not worth it.")

chair_content = instructions.ADD_ACCESSOR("They are talking about chairs")
instructions.ADD_CONTENT(chair_content, "Raphael likes his current chair")
instructions.ADD_CONTENT(chair_content, "Raphael bought his chair from ikea")

DTIME.lap("GETTING CONTENT")


content = instructions.FIND_CONTENT("They are talking about construction.")
print("FOUND CONTENT", "\n".join(content))


DTIME.lap("GETTING CONTENT")


content = instructions.FIND_CONTENT("This is about ikea.")
print("FOUND CONTENT", "\n".join(content))


# Extract all the content nodes and display what they are linking to
print("\nCONTENTS")
res = instructions.contents.get(include=["metadatas", "documents"])
for id_, metadata, document in zip(res["ids"], res["metadatas"], res["documents"]):
    print(id_, metadata, document)

print("\n\nACCESSORS")
res = instructions.accessors.get(include=["metadatas", "documents"])
for id_, metadata, document in zip(res["ids"], res["metadatas"], res["documents"]):
    print(id_, metadata, document)


DTIME.lap("DISPLAYING GRAPH")

from .plotly_display import display_plotly
display_plotly(instructions)
