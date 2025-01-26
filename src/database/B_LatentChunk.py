
from sqlalchemy import (
    Column, 
    Integer, 
    String, 
    ForeignKey,
)

from .naming import LATENT_CHUNKS_TABLE_NAME, CONTENT_TABLE_NAME
from .A_Edge import *


class LatentChunk(Base):
    """
    When many text inputs fall in the similarity_condition and not the identity_condition
    of a content node, they are stored in this table in reference to the node they were most similar with.

    When they accumulate for a particular content node, they are merged together using a LLM function
    and replace the text and vector of the content node, effectively replacing it with "more relevant" content.
    """
    __tablename__ = LATENT_CHUNKS_TABLE_NAME
    
    # Most similar content id
    id = Column(Integer, primary_key=True, autoincrement=True)
    content_id = Column('content_id', Integer, ForeignKey(f'{CONTENT_TABLE_NAME}.id', ondelete="CASCADE"), nullable=False)
    text = Column('text', String, nullable=False)
