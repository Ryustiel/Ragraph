"""
Define the accessor layers for the database.
Each accessor is defined by a label and a function that creates the content from the input.

NOTE : ACCESSOR_CONFIG is converted into a LAYERS by the .contents.py module which builds the database dynamically.
The AccessorConfig object is still accessible from the ORM.accessor_config attribute.
"""
import numpy

from pydantic import BaseModel
from typing import (
    List,
    Dict,
    Literal,
    Callable,
)

from ._vectorizer import Vectorizer

# ================================================================= TYPES

EmbeddingsVector = List[float]

class AccessorConfig(BaseModel):
    """
    Accessor layer definition data, to be converted to the actual layers. (Accessor Nodes)
    A layer is a set of accessors.

    NOTE : This Metadata => Accessor conversion step is ONLY here for readability.
    I couldve also just defined the tables directly but this structure is much easier to review.
    """
    embeddings_function: Callable[[str], EmbeddingsVector]
    embeddings_dimension: int

    type: Literal["text", "keywords"]

    similarity_condition: Callable[[EmbeddingsVector, EmbeddingsVector], bool]  # How different a vector can be from an accessor to still be matched
    shift_factor: float = 0 # How fast the accessor will move in the vector space towards each match on a positive content update

# ================================================================= BASE FUNCTIONS

def l2_norm(less_than: float):
    def f(x: List[float], y: List[float]) -> bool:
        res = numpy.linalg.norm(x - y)
        return res < less_than
    return f

# ================================================================= PARAMETERS

# Determines when a chunk si similar enough that it's relevant to merge
CONTENT_SIMILARITY_CONDITION: Callable[[EmbeddingsVector, EmbeddingsVector], bool] = l2_norm(0.5)

# Determines when a chunk is too similar to be worth merging
CONTENT_IDENTITY_CONDITION: Callable[[EmbeddingsVector, EmbeddingsVector], bool] = l2_norm(0.1)

# Content embedding config
CONTENT_EMBEDDINGS_FUNCTION: Callable[[str], EmbeddingsVector] = Vectorizer.process
CONTENT_EMBEDDINGS_DIMENSIONS: int = 3072

# Accessors
ACCESSOR_CONFIG: Dict[str, AccessorConfig] = {

    "test": AccessorConfig(
        type = "text",
        similarity_condition = l2_norm(0.5),
        embeddings_function = Vectorizer.process,
        embeddings_dimension = 3072,
    ),

    "words": AccessorConfig(
        type = "keywords",
        similarity_condition = l2_norm(0.5),
        embeddings_function = Vectorizer.process,
        embeddings_dimension = 3072,
    ),

}
