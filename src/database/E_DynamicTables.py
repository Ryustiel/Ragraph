"""
Builds the accessor and edge tables dynamically
from the base classes and the config metadata.
"""

from typing import (
    Dict,
)
from sqlalchemy import Column, Integer, String, ForeignKey
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship

from ._config import ACCESSOR_CONFIG, AccessorConfig
from .naming import *
from .D_Accessor import *


ACCESSORS: Dict[str, Accessor] = {}

for layer_name, config in ACCESSOR_CONFIG.items():
    
    accessor_edges = type(
        edge_class_name(layer_name),
        (AccessorEdge,),
        {
            "__tablename__": edge_table_name(layer_name),
            "accessor_id": Column(
                'accessor_id', 
                Integer, 
                ForeignKey(f'{accessor_table_name(layer_name)}.id'), 
                primary_key=True,
            ),
            "content": relationship(
                "Content", 
                backref=attribute_name(layer_name),
            ),
            "accessor": relationship(
                accessor_class_name(layer_name),
                backref="contents",
            ),
        },
    )
    
    accessor_class = type(
        accessor_class_name(layer_name),
        (Accessor,),
        {
            "__tablename__": accessor_table_name(layer_name),
            "embedding": Column(
                Vector(dim=config.embeddings_dimension),
                nullable=False,
            ),
            "edges": accessor_edges,
            "layer_name": layer_name,
            "accessor_config": config, 
        },
    )

    ACCESSORS[layer_name] = accessor_class
