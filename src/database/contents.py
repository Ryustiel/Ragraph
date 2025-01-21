"""
Context graphs represent structured information, organized by contexts in various layers.

On single "content" layer is linked and accessed by multiple "accessor" layers, via weighted links.
Contents can be linked to one another when they represent very similar but very dense information.
Contents also come with some metadata that may impact their "relevancy" score and make them come up more frequently. (for example, whether a content is an instruction or not.)

Each layer has its own family of accessor nodes which represent a different "characterization" of context.
For example, a context layer can represent a "who the participant in the conversation are" type of context.

Together, the many layers of accessor and their varying connection to contents represent relevancy of information.

[A complex set of Graph-Vector Store layers]
"""

from typing import List, Dict

from sqlalchemy import (
    Column, Integer, String, 
    Float, ForeignKey, Table, 
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.declarative import declared_attr

from ._accessor_config import LAYER_DATA

Base = declarative_base()
ACCESSORS: Dict[str, Table] = {}

def table_name_from_label(label: str) -> str:
    return f"context_{label}"

# Many-to-Many association table for the graph neighbor slot
# NOTE : There should be one for each accessor layer + 1 for the content to content relationship
content_edges = Table(
    "context_contents_edges",
    Base.metadata,
    Column("node_id", Integer, ForeignKey("context_contents.id"), primary_key=True),
    Column("neighbor_id", Integer, ForeignKey("context_contents.id"), primary_key=True),
)

class Content(Base):
    """
    The content nodes contain a piece of information in the form of text.
    """
    __tablename__ = "context_contents"

    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)   # The actual content, meaningful piece of information
    embedding = Column(Vector(dim=3072))   # Vector slot for the content

    neighbors = relationship(   # Handles content to content neighbors
        "Content",
        secondary = content_edges,
        primaryjoin = id == content_edges.c.node_id,
        secondaryjoin = id == content_edges.c.neighbor_id,
        backref = "linked_by",  # Adds a reverse relationship
    )

class Accessor(Base):
    """
    This base class for accessors.
    This one is abstract and needs to be subclassed as is below.
    """
    __abstract__ = True

    id = Column(Integer, primary_key=True)

    @declared_attr
    def embedding(cls):
        """
        This is a placeholder for the embedding attribute,
        which will be of varying length depending on the accessor class.
        """
        return Column(Vector(dim=0)) # Vector slot for the content

    @declared_attr
    def contents(cls):
        """
        This is a placeholder for the contents relationship.
        The actual table will resolve dynamically for subclasses.
        """
        return relationship(
            "Content",
            secondary=lambda: None,  # Placeholder: real table will be set in dynamically created classes
            primaryjoin=lambda: False,  # Prevents eager resolving
            viewonly=True,  # Placeholder relationship
            doc="This is a placeholder for content relationships defined in subclasses.",
        )

# ================================================================= ACCESSORS


for layer_name, data in LAYER_DATA.items():
    
    # Create an accessor table from the metadata
    accessor_edges = Table(
        f'context_{layer_name}_edges',
        Base.metadata,
        Column('node_id', Integer, ForeignKey(f'context_{layer_name}.id'), primary_key=True),
        Column('neighbor_id', Integer, ForeignKey('context_contents.id'), primary_key=True),
    )
    
    accessor_class = type(
        f"Context{layer_name.capitalize()}Accessor",
        (Accessor,),  # Inherit from Base
        {
            "__tablename__": table_name_from_label(layer_name),  # Table name
            "embedding": Column(Vector(dim=data.embeddings_dimension)),

            # Define the relationship to the content nodes
            "contents": relationship(
                "Content",
                secondary = accessor_edges,
                primaryjoin = f"context_{layer_name}.id == {accessor_edges}.c.node_id",
                secondaryjoin = f"Content.id == {accessor_edges}.c.neighbor_id",
                backref=table_name_from_label(layer_name),  # Backrefs in Content are refered to as the table name itself
            ),
        },
    )

    ACCESSORS[layer_name] = accessor_class
