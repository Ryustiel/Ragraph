
from typing import (
    List, 
    Tuple,
    ClassVar,
    Optional,
)
from sqlalchemy import Column, Integer
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Session, Mapped
from sqlalchemy.ext.declarative import declared_attr

from .config import LayerConfig
from ._node_set import NodeSet
from ._B_Content import *


class Accessor(Node):
    """
    This base class for accessors.
    This one is abstract and needs to be subclassed as is below.
    """
    __abstract__ = True

    EdgeORM = ClassVar[Edge]
    layer_name = ClassVar[str]
    layer_config = ClassVar[LayerConfig]

    embeddable: Mapped[str] = Column('embeddable', String, nullable=False)

    @declared_attr
    def embedding(cls) -> Mapped[Vector]:
        """
        An embeddings vector of varying length depending on the accessor class.
        Subclasses must override this attribute.
        """
        raise NotImplementedError("Subclasses must define the 'embedding' column")

    @declared_attr
    def proxy(cls) -> Mapped[Optional[int]]:
        """
        A foreign key to an accessor row in the same layer / table.
        Subclasses must override this attribute.
        """
        raise NotImplementedError("Subclasses must define the 'proxy' column")
     

    def next_contents(self, session: Session) -> NodeSet[Content]:
        """
        Returns a set of weighted contents from the edges of this accessor (or the accessor it's a proxy of).
        """
        if self.proxy is not None:
            id = self.proxy
        else:
            id = self.id

        edges = session.query(self.EdgeORM).filter(self.EdgeORM.accessor_id == id).all()
        return NodeSet[Content]({ edge.content: (edge.weight, 1) for edge in edges })
    
    
    def update_vector(self):
        """
        Update the embeddings vector of this class based on the current "embeddable" string.
        """
        self.embedding = self.layer_config.embeddings_function(self.embeddable)


    @classmethod
    def vector_search(cls, session: Session, input: str, max_output: int = 1) -> NodeSet['Accessor']:
        """
        A simple vector search on one input.
        """
        vector = cls.layer_config.embeddings_function(input)

        subq = (
            session.query(
                cls.id.label("id"),
                cls.embedding.cosine_distance(vector).label("similarity")
            )
            .subquery()
        )

        # Join back to the full table, so we load the entire accessor row.
        q = (
            session.query(cls, subq.c.similarity)
            .join(subq, cls.id == subq.c.id)
            .order_by(subq.c.similarity)
            .limit(max_output)
        )

        accessor_set = NodeSet[Accessor]()

        for accessor, similarity in q.all():
            # Check if this accessor is a proxy for another
            if accessor.proxy is not None:
                if accessor.proxy in accessor_set.keys():
                    # A better proxy has been added previously; skip
                    continue
                else:
                    accessor = accessor.proxy

            accessor_set[accessor] = (similarity, 1)

        return accessor_set
