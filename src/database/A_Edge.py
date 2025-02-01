
from typing import (
    List,
    Dict,
    Optional,
    Literal,
    TypedDict,
    Type,
)
from datetime import timedelta, datetime

from sqlalchemy import (
    Column, Integer, DateTime, JSON,
    Float, ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship, Session
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.sql import exists

from ._naming import CONTENT_TABLE_NAME, CONTENT_LINK_TABLE_NAME, LATENT_REVIEW_TABLE_NAME

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



class LatentReviewMessage(TypedDict):
    type: Literal["human", "ai", "system"]
    message: str

class LatentReview(Base):
    """
    LatentReview are objects that represent bits of a conversation linked to the edges that were activated during chunk retrieval.
    LLM Functions should be called on the contents of this review in order to do updates to the database.

    Reviews will store chunks of a conversation. Once enough chunks are stored or the max delay between messages is reached, 
    LLM functions will be called to extract information or to assess qualities of the contents that are retrieved from the selected edges.
    The weight of those edges will be bulk updated depending on the quality of the reviews and the weight they already have. (slower decrease for lower weight relative to all the edges that brought up reviews according to a set accessor context)

    This means : for each accessor in edges, compute the max weight and then distance to max weight for each edge.
    Then when processing, we're focusing on one content one by one. When updating the weight of the associated edge the formula takes into account
    1. The llm review (influences the strength and direction of the update)
    2. The distance to the max weight among the other content accessed by the accessor at the time (influences the strength of the update only)

    Also include in the prompt sequence a prompt to "disqualify informaton" and to "qualify or require information" that are a single post process step on the whole conversation.
    Give both of them the knowledge that's available so that they avoid redundancies. They should also be able to make edits to the knowledge (with correcting notes) => Gets added to latent chunks
    """
    __tablename__ = LATENT_REVIEW_TABLE_NAME

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(JSON, nullable=False)
    last_updated = Column(DateTime, nullable=False)

    # relationships with the edges here through the association table

    def add_message(self, type: Literal["human", "ai", "system"], message: str):
        """
        Add a message to the conversation JSON, and more edges.
        """
        pass

    def add_edges(self, edges: Dict[Type[AccessorEdge], List[AccessorEdge]]):
        """
        Add more edges to the conversation.
        """
        pass

    def has_expired(self, session: Session, expiration_threshold: timedelta) -> bool:
        """
        Check if the review has expired (time has passed since the last update).
        If a review expires but was not completed, extract some information (new contents)
        and then delete the review.
        """
        pass

    @classmethod
    def create_review(cls, session: Session, messages: List[LatentReviewMessage], edges: Dict[Type[AccessorEdge], List[AccessorEdge]]):
        """
        Create a new review and add messages to it.
        """
        pass

    @classmethod
    def delete_outdated_incomplete_reviews(cls, session: Session, outdated_threshold: timedelta):
        """
        Delete all reviews that have "time" expired without being completed.
        """
        pass
