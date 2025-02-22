
from typing import (
    List,
)

from sqlalchemy import (
    Column, Integer,
    Float, ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship, Session, Mapped
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.sql import exists

from .naming import CONTENT_TABLE_NAME

Base = declarative_base()


class Node(Base):
    __abstract__ = True  # No table will be created for Node
    
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    
    def __hash__(self):
        """Unique hash for indexing in a Set."""
        return hash(self.id)


class Edge(Base):
    """
    This is a base class for edges between accessors and contents.
    This one is abstract and needs to be subclassed as is below.
    """
    __abstract__ = True

    weight: Mapped[float] = Column('weight', Float, nullable=False, default=0.5)
    content_id: Mapped[int] = Column('content_id', Integer, ForeignKey(f'{CONTENT_TABLE_NAME}.id'), primary_key=True)

    @declared_attr
    def accessor_id(cls) -> Mapped[int]:
        """The id of an accessor"""
        raise NotImplementedError("Subclasses must define the 'accessor_id' column")

    @declared_attr
    def accessor(cls):
        """The accessor this edge connects to"""
        pass
    
    @declared_attr
    def content(cls):
        """The content this edge connects to"""
        pass

    @classmethod
    def exists(cls, session: Session, content_id: int, accessor_id: int) -> bool:
        return session.query(exists().where(
            cls.content_id == content_id,
            cls.accessor_id == accessor_id
        )).scalar()


