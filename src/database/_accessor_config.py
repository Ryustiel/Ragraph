"""
Define the accessor layers for the database.

Each accessor is defined by a label and a function that creates the content from the input. Callable

NOTE : ACCESSOR_CONFIG is converted into a LAYERS by the .contents.py module which builds the database dynamically.
The AccessorConfig object is still accessible from the ORM.accessor_config attribute.
"""
from pydantic import BaseModel
from typing import (
    List,
    Dict,
    Any,
    Literal,
    Tuple,
    Callable,
    Union,
    Optional,
)

from ._vectorizer import Vectorizer

# ================================================================= TYPES

EmbeddingsVector = List[float]

class BaseContextInput(BaseModel):
    """
    Lets you define the current context and extract content nodes from the various accessor layers based on it.
    """
    context: Dict[str, Union[str, List[str]]]

class AccessorConfig(BaseModel):
    """
    Accessor layer definition data, to be converted to the actual layers. (Accessor Nodes)
    A layer is a set of accessors.

    NOTE : This Metadata => Accessor conversion step is ONLY here for readability.
    I couldve also just defined the tables directly but this structure is much easier to review.
    """
    embeddings_function: Callable[[Any], EmbeddingsVector]
    embeddings_dimension: int

    type: Literal["text", "keywords"]

    similarity_threshold: float  # How different a vector can be from an accessor to still be matched
    shift_factor: float = 0 # How fast the accessor will move in the vector space towards each match

    def get_input_type(self):
        if self.type in ("text", ):
            return str
        elif self.type in ("keywords", ):
            return list

# ================================================================= VECTORIZER FUNCTIONS

# Functions that will be used to convert input data to vectors

# ================================================================= ACCESSORS

ACCESSOR_CONFIG: Dict[str, AccessorConfig] = {

    "test": AccessorConfig(
        type = "text",
        similarity_threshold = 0.9,
        embeddings_function = Vectorizer.process,
        embeddings_dimension = 3072,
    ),

    "words": AccessorConfig(
        type = "keywords",
        similarity_threshold = 0.9,
        embeddings_function = Vectorizer.process,
        embeddings_dimension = 3072,
    ),

}
