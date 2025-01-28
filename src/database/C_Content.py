
from typing import (
    List, 
    Dict,
    Any,
    Tuple,
    Generator,
    Union,
)

from sqlalchemy import (
    Column, Integer, String, 
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship, declarative_base, Session

from ._config import CONTENT_EMBEDDINGS_FUNCTION, CONTENT_EMBEDDINGS_DIMENSIONS
from ._vectorizer import Vectorizer
from ._naming import CONTENT_TABLE_NAME, CLUSTER_TABLE_NAME, ACCESSOR_ATTRIBUTE_PREFIX
from .B_Cluster import *


# ================================================================= CONTENT SET (DATA TRANSPORT)


class ContentSet(Dict['Content', float]):
    """
    A set of content nodes. This class is used to implement repetitive operations on sets of nodes.
    Keys are the ids of the contents in the database, and values are their associated weights.
    """
    def __add__(self, other: 'ContentSet') -> 'ContentSet':
        """
        Summing two content sets results in a new set with the union of contents, 
        with new weights representing the combination of both.
        """
        result = self

        for other_key in other.keys():
            if other_key in self.keys():
                result[other_key] = max(self[other_key], other[other_key])
            else:
                result[other_key] = other[other_key]

        return result

    def __mul__(self, factor: float) -> 'ContentSet':
        result = ContentSet()
        for key, weight in self.items():
            result[key] = weight * factor
        return result


# ================================================================= CONTENT ORM


class Content(Base):
    """Represents a piece of information that can be accessed through accessors."""

    __tablename__ = CONTENT_TABLE_NAME
    
    # TODO : Use Mapped for typing in here.

    id = Column('id', Integer, primary_key=True, autoincrement=True)
    text = Column('text', String, nullable=False)
    embedding = Column('embedding', Vector(dim=CONTENT_EMBEDDINGS_DIMENSIONS), nullable=False)
    behavior = Column('behavior', String(20), nullable=False, default='regular')  # Special flag for the content, has no use for now
    cluster_id = Column('cluster_id', Integer, ForeignKey(f"{CLUSTER_TABLE_NAME}.id"), nullable=True, default=None)

    cluster = relationship('Cluster', back_populates="contents")
    latent_chunks = relationship('LatentChunk', back_populates="content")

    outgoing_links = relationship(
        "ContentLink",
        foreign_keys="[ContentLink.origin]",
        back_populates="origin_content",
    )
    incoming_links = relationship(
        "ContentLink",
        foreign_keys="[ContentLink.target]",
        back_populates="target_content",
    )
    
    # Convenience property to get neighbors with weights
    @property
    def neighbors(self):
        return [(link.target_content, link.weight) for link in self.outgoing_links]


    def __hash__(self):
        """Unique hash for indexing in a ContentSet"""
        return self.id


    # ================================================================= VECTOR SEARCH


    @classmethod
    def find_content(cls, session: Session, input: Union[str, float], max_output: int = 10) -> List['Content']:
        """
        Runs a vector search for content similar to the current one.
        """
        if isinstance(input, str):
            input = CONTENT_EMBEDDINGS_FUNCTION(input)
        return (
            session.query(cls)
            .filter(cls.embedding != None)  # Exclude entries without vectors
            .order_by(cls.embedding.l2_distance(input))  # L2 distance (Euclidean distance)  NOTE : If this must be changed, also change the Context methods at .interface.py
            .limit(max_output)
            .all()
        )
    

# ================================================================= COMPUTE 2ND DEGREE CONTENTS


    @property
    def linked_accessors(self) -> Generator[List[AccessorEdge], Any, None]:
        """The list of accessors that were linked to the current content."""
        # look at all the attributes whose name begin with ACCESSOR_ATTRIBUTE_PREFIX
        # yield the values of the attribute
        for attribute_name, value in self.__dict__.items():
            if attribute_name.startswith(ACCESSOR_ATTRIBUTE_PREFIX):
                yield value  # If an accessor attribute, then it contains the list of links to these accessors

    # NOTE : Contents should always be added via an accessor, not the other way around. No "accessor append" in this class.


    def through_weight_formula(self, link_weight_to_acessor: float, link_weight_to_content: float) -> float:
        """Calculate the weight of a content accessed through / via an intermediate accessor"""
        return link_weight_to_acessor * link_weight_to_content
    

    # TODO : Find related accessors + weights
    def next_contents(self) -> ContentSet:
        """
        Find the content nodes that can be reached indirectly from the current content node.

        1. Look for the accessors linked to the current content and retain the weight of the links
        2. Get the contents attached to each of these accessors and compute a new weight link 
        based on the weights of both the [current -> intermediate accessor] and [intermediate -> other content]
        3. Compile the weights of each accessor in a ContentSet
        4. Merge the content sets (which also retains the max weight of each accessor)
        5. Return the merged content set, effectively representing the content nodes that can be reached indirectly from the current content node.
        """
        sets = []
        for links in self.linked_accessors:
            for link in links:
                local_set = ContentSet()
                for content_link in link.accessor.contents:
                    local_set[content_link.content.id] = self.through_weight_formula(link.weight, content_link.weight)
                sets.append(local_set)
        return sum(sets) if sets else ContentSet()


    # ================================================================= DISPLAY


    @classmethod
    def get_nodes(cls, session: Session) -> List[Tuple[int, str]]:
        contents = session.query(cls).all()
        return [(content.id, content.text) for content in contents]
