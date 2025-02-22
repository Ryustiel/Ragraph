"""
Define the accessor layers for the database.
Each accessor is defined by a label and a function that creates the content from the input.

NOTE : LAYER_CONFIG is converted into a LAYERS by the .contents.py module which builds the database dynamically.
The LayerConfig object is still accessible from the ORM.LAYER_CONFIG attribute.
"""
import numpy

from pydantic import BaseModel
from typing import (
    get_args,
    List,
    Dict,
    Set,
    Literal,
    Callable,
)

from ._vectorizer import Vectorizer

# ================================================================= TYPES

EmbeddingsVector = List[float]

class LayerConfig(BaseModel):
    """
    Accessor layer definition data, to be converted to the actual layers. (Accessor Nodes)
    A layer is a set of accessors.

    NOTE : This Metadata => Accessor conversion step is ONLY here for readability.
    I couldve also just defined the tables directly but this structure is much easier to review.
    """
    embeddings_function: Callable[[str], EmbeddingsVector]
    embeddings_dimension: int

    type: Literal["text"]

    similarity_condition: Callable[[EmbeddingsVector, EmbeddingsVector], bool]  # How different a vector can be from an accessor to still be matched

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

def THROUGH_WEIGHT_FORMULA(weight_from_origin: float, weight_to_destination: float) -> float:
    """
    Calculate the weight of a destination content, 
    accessed from an origin content, 
    through an intermediate accessor.
    """
    return weight_from_origin * weight_to_destination

# Accessors
LAYER_CONFIG: Dict[str, LayerConfig] = {

    "tasks": LayerConfig(
        type = "text",
        similarity_condition = l2_norm(0.5),
        snap_condition = l2_norm(0.8),
        embeddings_function = Vectorizer.process,
        embeddings_dimension = 3072,
    ),

    "qna": LayerConfig(
        type = "text",
        similarity_condition = l2_norm(0.5),
        snap_condition = l2_norm(0.8),
        embeddings_function = Vectorizer.process,
        embeddings_dimension = 3072,
    ),

    "topics": LayerConfig(
        type = "text",
        similarity_condition = l2_norm(0.5),
        snap_condition = l2_norm(0.8),
        embeddings_function = Vectorizer.process,
        embeddings_dimension = 3072,
    ),

    "people": LayerConfig(
        type = "text",
        similarity_condition = l2_norm(0.5),
        snap_condition = l2_norm(0.8),
        embeddings_function = Vectorizer.process,
        embeddings_dimension = 3072,
    ),

}

LayerName = Literal["tasks", "qna", "topics", "people"]

ALL_LAYER_NAMES: Set[str] = set(get_args(LayerName))

# Runtime check: ensure all keys in LAYER_CONFIG are valid allowed layers
invalid_keys = ALL_LAYER_NAMES.symmetric_difference(set(LAYER_CONFIG.keys()))
if invalid_keys:
    raise ValueError(f"LAYER_CONFIG contains keys you did not register in \"database/config.py/LayerName\": {invalid_keys}")
