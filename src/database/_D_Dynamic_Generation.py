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

from .config import LAYER_CONFIG
from .naming import *
from ._C_Accessor import *


ACCESSORS: Dict[str, Accessor] = {}

for layer_name, layer_config in LAYER_CONFIG.items():
    
    LayerEdgeORM = type(
        edge_class_name(layer_name),
        (Edge,),
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

    LayerAccessorORM = type(
        accessor_class_name(layer_name),
        (Accessor,),
        {
            "__tablename__": accessor_table_name(layer_name),
            "embedding": Column(
                Vector(dim=layer_config.embeddings_dimension),
                nullable=False,
            ),
            "proxy": Column(
                Integer,
                ForeignKey(f'{accessor_table_name(layer_name)}.id')
            ),
            "EdgeORM": LayerEdgeORM,
            "layer_name": layer_name,
            "layer_config": layer_config, 
        },
    )

    ACCESSORS[layer_name] = LayerAccessorORM
