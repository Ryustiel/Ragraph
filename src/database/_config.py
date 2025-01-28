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
    snap_condition: Callable[[EmbeddingsVector, EmbeddingsVector], bool] # When a new accessor is added to this layer, 
    # the snap condition is tested against the most similar accessor in the layer. If it is met, 
    # the current edges of the similar, existing accessor will be added to the new accessor so that it already has "relevant" content nodes linked, as a starting point.
    # It can potentially lose those edges over time via the weight evaluation process, if they were not actually relevant.
    # NOTE : Ths process only happens to new accessor nodes when they are first added.
    shift_factor: float = 0 # How fast the accessor will move in the vector space towards each match on a positive content update

# ================================================================= BASE FUNCTIONS

def l2_norm(less_than: float):
    def f(x: List[float], y: List[float]) -> bool:
        res = numpy.linalg.norm(x - y)
        return res < less_than
    return f

# ================================================================= PARAMETERS

EDGE_DELETE_CONDITION: Callable[[float], bool] = lambda x: x < 0.1

# Weight update formulas
WEIGHT_STRONG_INCREASE: Callable[[float], float] = lambda x: min(x * 1.5, 1)
WEIGHT_INCREASE: Callable[[float], float] = lambda x: min(x * 1.1, 1)
WEIGHT_DECREASE: Callable[[float], float] = lambda x: max(x * 0.9, 0)
WEIGHT_STRONG_DECREASE: Callable[[float], float] = lambda x: max(x * 0.5, 0)

# Content Links - "former" is the former weight, x is the new weight 
BREAK_CLUTER_CONDITION: Callable[[float, float], bool] = lambda x, former: x < former and x < 0.4 
CREATE_CLUTER_CONDITION: Callable[[float, float], bool] = lambda x, former: x > former and x > 0.6

# Content link will be deleted when below this threshold to save space in the database
CONTENT_LINK_DELETE_CONDITION: Callable[[float], bool] = lambda x: x < 0.2

# Link weight update formulas
CONTENT_LINK_STRONG_INCREASE: Callable[[float], bool] = lambda x: x * 1.5
CONTENT_LINK_INCREASE: Callable[[float], bool] = lambda x: x * 1.1
CONTENT_LINK_DECREASE: Callable[[float], bool] = lambda x: x * 0.9
CONTENT_LINK_STRONG_DECREASE: Callable[[float], bool] = lambda x: x * 0.5

# Determines when a chunk si similar enough that it's relevant to merge
CONTENT_SIMILARITY_CONDITION: Callable[[EmbeddingsVector, EmbeddingsVector], bool] = l2_norm(0.5)
# Determines when a chunk is too similar to be worth merging
CONTENT_IDENTITY_CONDITION: Callable[[EmbeddingsVector, EmbeddingsVector], bool] = l2_norm(0.1)
# Content embedding config
CONTENT_EMBEDDINGS_FUNCTION: Callable[[str], EmbeddingsVector] = Vectorizer.process
CONTENT_EMBEDDINGS_DIMENSIONS: int = 3072

# Accessors
ACCESSOR_CONFIG: Dict[str, AccessorConfig] = {

    "tone": AccessorConfig(
        type = "text",
        similarity_condition = l2_norm(0.5),
        snap_condition = l2_norm(0.8),
        embeddings_function = Vectorizer.process,
        embeddings_dimension = 3072,
    ),

    "user_query": AccessorConfig(
        type = "text",
        similarity_condition = l2_norm(0.5),
        snap_condition = l2_norm(0.8),
        embeddings_function = Vectorizer.process,
        embeddings_dimension = 3072,
    ),

    "action": AccessorConfig(
        type = "text",
        similarity_condition = l2_norm(0.5),
        snap_condition = l2_norm(0.8),
        embeddings_function = Vectorizer.process,
        embeddings_dimension = 3072,
    ),

    "feasability": AccessorConfig(
        type = "text",
        similarity_condition = l2_norm(0.5),
        snap_condition = l2_norm(0.8),
        embeddings_function = Vectorizer.process,
        embeddings_dimension = 3072,
    ),

}
