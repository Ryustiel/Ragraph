
from typing import (
    List, 
    ClassVar,
    Optional,
)
from sqlalchemy import Column, Integer
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declared_attr

from ._config import AccessorConfig
from .C_Content import *



class Accessor(Base):
    """
    This base class for accessors.
    This one is abstract and needs to be subclassed as is below.
    """
    __abstract__ = True

    edges = ClassVar[AccessorEdge]
    layer_name = ClassVar[str]
    accessor_config = ClassVar[AccessorConfig]

    id = Column(Integer, primary_key=True, autoincrement=True)

    @declared_attr
    def embedding(cls):
        """
        This is a placeholder for the embedding attribute,
        which will be of varying length depending on the accessor class.
        """
        return Column(Vector(dim=0)) # Vector slot for the content
     

    # ================================================================= COMPUTE NEXT CONTENTS


    def get_contents(self, session: Session) -> ContentSet:
        """
        Returns a set of weighted contents from the edges of this accessor.
        """
        contents = ContentSet()
        edges = session.query(self.edges).filter(self.edges.accessor_id == self.id).all()
        for edge in edges:
            contents[edge.content] = edge.weight
        return contents

    @classmethod
    def from_text(cls, session: Session, text: str) -> 'Accessor':
        """
        GET OR CREATE

        1. Finds the most similar accessor in the vector db.
        2. If the closest match triggers the SIMILARITY CONDITION, return the accessor.
        3. Otherwise, create a new accessor in the layer with the input text as a content.
        4. If the new accessor triggers the SNAP CONDITION with the closest match, 
        duplicate the edges of the closest match onto the new accessor. (so that it already starts with a couple edges and weights set)
        """
        # Find the most similar accessor in the vector db
        similar_accessors = cls.vector_search(session, cls.accessor_config.embeddings_function(text), max_output=1)
        
        if similar_accessors:
            # If the closest match is below the similarity threshold, return the accessor
            most_similar_accessor = similar_accessors[0]
            if cls.accessor_config.similarity_condition(most_similar_accessor.embedding, cls.accessor_config.embeddings_function(text)):
                return most_similar_accessor

        # Otherwise, create a new accessor in the layer with the input text as a content
        new_embedding = cls.accessor_config.embeddings_function(text)
        new_accessor = cls(embedding=new_embedding)
        
        # Use the snap condition to add edges pointing to the same content nodes
        if similar_accessors:
            most_similar_accessor = similar_accessors[0]
            # Only add the edges if the snap condition is True.
            if cls.accessor_config.snap_condition(new_embedding, most_similar_accessor.embedding):
                similar_edges = session.query(cls.edges).filter(cls.edges.accessor_id == most_similar_accessor.id).all()
                for edge in similar_edges:
                    new_edge = cls.edges(accessor_id=new_accessor.id, content_id=edge.content_id, weight=edge.weight)
                    session.add(new_edge)
        
        session.add(new_accessor)
        return new_accessor
    
    @classmethod
    def from_id(cls, session: Session, id: int) -> Optional['Accessor']:
        """
        GET

        Fetches an accessor from the database by its ID.
        """
        return session.query(cls).filter(cls.id == id).first()

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


    # ================================================================= DISPLAY

    @classmethod
    def get_nodes(cls, session: Session) -> List[Tuple[int, str]]:
        accessors = session.query(cls).all()
        return [(accessor.id, "") for accessor in accessors]

    @classmethod
    def get_edges(cls, session: Session) -> List[Tuple[int, int, float]]:
        edges = session.query(cls.edges).all()
        return [(edge.accessor_id, edge.content_id, edge.weight) for edge in edges]

    @classmethod
    def find_node(cls, session: Session, text: str) -> Optional[int]:
        """
        Find an accessor node from this layer given a content input.
        """
        # Find the most similar accessor in the vector db
        similar_accessors = cls.vector_search(session, cls.accessor_config.embeddings_function(text), max_output=1)
        
        if similar_accessors:
            # If the closest match is below the similarity threshold, return the accessor
            most_similar_accessor = similar_accessors[0]
            return most_similar_accessor.id
