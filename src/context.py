"""
Define the interfacing models, that accept outside information (the context) or produce exportable models.
Most of these models must be pydantic because they will be interacted with by the FastAPI layer.
"""
from typing import (
    Dict,
    List,
)
from sqlalchemy.orm import Session

from .database import (
    Content, 
    Accessor, 
    ContentSet, 
    ACCESSORS, 
    EmbeddingsVector,
)


class ContextInput(Dict[str, str]):
    """
    Lets you define the current context and extract content nodes from the various accessor layers based on it.
    """

    def compute_vectors(self) -> Dict[str, EmbeddingsVector]:
        """
        Compute the vectors for each accessor from the input.

        vectors = context.compute_vectors()  # for vizualisation
        for space, vect in vectors.items():
            print(f"\t{space}: {vect[0]} {len(vect)}")
        """
        vectors = {}

        for layer_name, layer_data in self.items():
            if layer_name not in ACCESSORS.keys():
                pass  # Skipping the parsing step for any layer that was not registered in context.accessors.py
            else:
                layer = ACCESSORS[layer_name].accessor_config
                vectors[layer_name] = layer.embeddings_function(layer_data)

        return vectors
    