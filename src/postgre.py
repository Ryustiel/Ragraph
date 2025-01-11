"""
Handle engine initialization, table creation etc.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .context import (
    ContextBase as ContextGraph,
    Content,
    ACCESSORS,
)
from .prompt import (
    PromptBase as PromptGraph,
    UserStatement,
    SuggestedReply,
)

def create_session(url: str) -> Session:
    """
    Create a new session and initializes 
    """
    engine = create_engine(url)

    # Initialize all the tables
    ContextGraph.metadata.create_all(engine)
    PromptGraph.metadata.create_all(engine)

    # Create and return the session instance
    Session = sessionmaker(bind=engine)
    return Session()
