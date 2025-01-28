
from typing import (
    Optional,
)

from sqlalchemy import (
    Column, Integer,
    Float, ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship, Session
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.sql import exists

from ._naming import CONTENT_TABLE_NAME, CONTENT_LINK_TABLE_NAME

Base = declarative_base()


class AccessorEdge(Base):
    """
    This is a base class for edges between accessors and contents.
    This one is abstract and needs to be subclassed as is below.
    """
    __abstract__ = True

    weight = Column('weight', Float, nullable=False, default=0.5)
    content_id = Column('content_id', Integer, ForeignKey(f'{CONTENT_TABLE_NAME}.id'), primary_key=True)

    @declared_attr
    def accessor_id(cls):
        """The id of an accessor"""
        return 0

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
    
    @classmethod
    def get_edge(cls, session: Session, content_id: int, accessor_id: int) -> Optional['AccessorEdge']:
        return session.query(cls).filter_by(content_id=content_id, accessor_id=accessor_id).first()


class ContentLink(Base):
    """
    Contents form a subgraph where the edges (called links) represent how often they end up called together. (Meaning in similar context sets)
    These "special edges" are used to propagate cluster markings for aggregating similar content together.
    """
    __tablename__ = CONTENT_LINK_TABLE_NAME

    origin = Column('origin', Integer, ForeignKey(f'{CONTENT_TABLE_NAME}.id'), nullable=False, primary_key=True)
    target = Column('target', Integer, ForeignKey(f'{CONTENT_TABLE_NAME}.id'), nullable=False, primary_key=True)
    weight = Column('weight', Float, nullable=False, default=0.5)

    origin_content = relationship(
        "Content",
        foreign_keys=[origin],
        back_populates="outgoing_links",
    )
    target_content = relationship(
        "Content",
        foreign_keys=[target],
        back_populates="incoming_links",
    )
