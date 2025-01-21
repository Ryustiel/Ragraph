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

from typing import (
    List, 
    Dict,
    ClassVar,
)

from sqlalchemy import (
    Column, Integer, String, 
    Float, ForeignKey, Table, 
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship, declarative_base, Session
from sqlalchemy.ext.declarative import declared_attr

from ._vectorizer import Vectorizer
from ._accessor_config import ACCESSOR_CONFIG, AccessorConfig

Base = declarative_base()
ACCESSORS: Dict[str, 'Accessor'] = {}

def to_accessor_table(label: str) -> str:
    """Generates a name for an accessor table"""
    return f"context_{label}"

def to_accessor_class(label: str) -> str:
    """Generates a name for a class that will be used to define a database model for an accessor"""
    return f"Context{layer_name.capitalize()}Accessor"

def to_edge_table(label: str) -> str:
    """Generates a name for an accessor edge table"""
    return f"{to_accessor_table(label)}_edges"

def to_edge_class(label: str) -> str:
    """Generates a name for a class that will be used to define a database model for the edges of an accessor"""
    return f"{to_accessor_class(label)}Edge"

class ContentSet(Dict):
    """
    A set of content nodes. This class is used to implement repetitive operations on sets of nodes.
    Keys are the ids of the contents in the database, and values are their associated weights.
    """
    def __sum__(self, other: 'ContentSet') -> 'ContentSet':
        """
        Summing two content sets results in a new set with the union of contents, 
        with new weights representing the combination of both.
        """
        for key, other_weight in other.items():
            self[key] = max(self[key], other_weight)

# ================================================================= ACTUAL DATABASE MODELS

class Content(Base):
    """
    The content nodes contain a piece of information in the form of text.
    """
    __tablename__ = "context__contents"

    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)   # The actual content, meaningful piece of information
    embedding = Column(Vector(dim=3072))   # Vector slot for the content
    behavior = Column(String(20), nullable=False, default="regular")

    @classmethod
    def find_content(cls, session: Session, content: str, max_output: int = 10) -> List['Content']:
        """
        Runs a vector search for content similar to the current one.
        """
        input_vector = Vectorizer.process(content)
        return (
            session.query(cls)
            .filter(cls.embedding != None)  # Exclude entries without vectors
            .order_by(cls.embedding.l2_distance(input_vector))  # L2 distance (Euclidean distance)
            .limit(max_output)
            .all()
        )

class AccessorEdge(Base):
    """
    This is a base class for edges between accessors and contents.
    This one is abstract and needs to be subclassed as is below.
    """
    __abstract__ = True

    Column('weight', Float, nullable=False, default=0.0),
    Column('content_id', Integer, ForeignKey('context__contents.id'), primary_key=True),

    @declared_attr
    def accessor_id(cls):
        """The id of an accessor"""
        pass

class Accessor(Base):
    """
    This base class for accessors.
    This one is abstract and needs to be subclassed as is below.
    """
    __abstract__ = True

    edges = ClassVar['AccessorEdge']
    layer_name = ClassVar[str]
    accessor_config = ClassVar[AccessorConfig]
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
    
    def get_contents(self) -> ContentSet:
        """
        Returns a set of weighted contents.
        """
        # TODO : Rever type the contents set file structure, converting ContentSet into a simple "summing methods" class.
        # TODO : Keep using both this and accessor sets. Compute a context vector when running the "save context" method.
        # Do we need to sum accessors at any point ? >> Fetching a list of accessors from each layer from context, only starting from one accessor and getting results; Echoing other accessors through context mapping; Linking new nodes to current contexts using ...
        # Yeah we absolutely don't need Accessor sets.
        # TODO : Delete accessors set once this is working
        pass
    
    @classmethod
    def vector_search(cls, session: Session, input_vector: List[float], max_output: int = 1) -> List['Accessor']:
        """
        Runs a vector search for content similar to the current one.
        """
        return (
            session.query(cls)
            .filter(cls.embedding != None)  # Exclude entries without vectors
            .order_by(cls.embedding.l2_distance(input_vector))  # L2 distance (Euclidean distance)
            .limit(max_output)
            .all()
        )

# ================================================================= DEFINING ACCESSOR SPECIALIZED ITEMS


for layer_name, config in ACCESSOR_CONFIG.items():
    
    accessor_edges = type(
        to_edge_class(layer_name),
        (AccessorEdge,),  # Inherit from Base
        {
            "__tablename__": to_edge_table(layer_name),  # Table name
            "accessor_id": Column('accessor_id', Integer, ForeignKey(f'{to_accessor_table(layer_name)}.id'), primary_key=True),
        },
    )
    
    accessor_class = type(
        to_accessor_class(layer_name),
        (Accessor,),  # Inherit from Base
        {
            "__tablename__": to_accessor_table(layer_name),  # Table name
            "embedding": Column(Vector(dim=config.embeddings_dimension)),

            # Define the relationship to the content nodes
            "contents": relationship(
                "Content",
                secondary = accessor_edges,
                primaryjoin = f"{to_accessor_class(layer_name)}.id == {to_edge_class(layer_name)}.c.accessor_id",
                secondaryjoin = f"Content.id == {accessor_edges}.c.content_id",
                backref=to_accessor_table(layer_name),  # Backrefs in Content are refered to as the table name itself
            ),

            "edges": accessor_edges,  # Set the accessor edges class to the accessor class variable for easy access.
            "layer_name": layer_name,
            "accessor_config": config, 
        },
    )

    ACCESSORS[layer_name] = accessor_class
