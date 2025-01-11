"""
Prompt databases store two main things : 

1. The exact (or compressed) sentences people said, indexed with a context set (multiple context vectors).
+ A context vector representing the kind of responses a LLM would have replied. (another set of context vectors)

2. Suggested replies that represent the "personnality" of the LLM, 
and the decisions it should typically make in particular contexts.
All that information is encoded in the replies OR system messages that can be registered as special "type 1. sentences".

This will be regularly queried after the context graph to get important components of the llm prompt.

[A simple 2 registries vector store like database]
"""

from sqlalchemy import (
    Column, Integer, String, 
    Float, ForeignKey, Table, 
    ARRAY,
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class UserStatement(Base):
    """
    Store sentences written by users. The sentences may be shortened.
    """
    __tablename__ = "prompt_user_statement"

    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)   # The stored user message
    context = Column(ARRAY(Float, dimensions=2))  # Set of vectors representing the context
    replies = Column(ARRAY(Float, dimensions=2))  # Set of vectors representing related responses (might be available in SuggestedReplies)

class SuggestedReply(Base):
    """
    Store suggested replies that will be used to guide the behavior of the LLM.
    """
    __tablename__ = "prompt_suggested_reply"

    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)  # The stored reply
    vector = (ARRAY(Float)) # Vector representation of text
