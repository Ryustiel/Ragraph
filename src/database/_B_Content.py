
from typing import (
    List, 
    Dict,
    Generator,
    Optional,
)

from sqlalchemy import (
    Column, String, 
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship, declarative_base, Session

from .config import (
    EmbeddingsVector,
    LayerName,
    THROUGH_WEIGHT_FORMULA,
    LAYER_CONFIG, 
)
from .naming import CONTENT_TABLE_NAME, attribute_name
from ._A_Edge_Node import *
from ._node_set import NodeSet


class Content(Node):
    """Represents a piece of information that can be accessed through accessors."""

    __tablename__ = CONTENT_TABLE_NAME

    text = Column('text', String, nullable=False)
    

    def get_linked_accessors(self, allowed_layers: Optional[List[LayerName]] = None) -> Generator[Edge, None, None]:
        """The list of accessors that were linked to the current content."""
        
        for accessor_layer_name in LAYER_CONFIG:

            if allowed_layers is not None and accessor_layer_name in allowed_layers:

                attr_name = attribute_name(accessor_layer_name)
                if hasattr(self, attr_name):
                    
                    # If an accessor attribute, then it contains the list of links to those accessors
                    for edge in getattr(self, attr_name):
                        if not isinstance(edge, Edge):
                            raise ValueError(f"get_linked_accessors() got an unexpected <{type(edge)}> during the process.")
                        yield edge


    def next_contents(self, allowed_layers: Optional[List[LayerName]] = None) -> NodeSet['Content']:
        """
        Parameters:
            * allowed_layers : the name of the accessor layers to include. Include all available if None.

        Find the content nodes that can be reached indirectly from the current content node.
        """
        next_contents = NodeSet[Content]()

        for edge in self.get_linked_accessors(allowed_layers=allowed_layers):

            edge_next_contents = NodeSet[Content]()

            through_edges: Generator[Edge, None, None] = edge.accessor.contents
            for through_edge in through_edges:  # Iterate over the contents accessible from this accessor

                if through_edge.content_id != edge.content_id:  # Exclude the content node that served as an access to the accessor

                    edge_next_contents[through_edge.content] = THROUGH_WEIGHT_FORMULA(edge.weight, through_edge.weight)
            
            next_contents = next_contents + edge_next_contents

        return next_contents
