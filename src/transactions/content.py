
from typing import (
    List,
    Self,
    Optional,
)
from .node_transaction import NodeTransaction
from sqlalchemy.orm import Session
from ..database import Content

from .context import Context


class ContentTransaction(NodeTransaction[Content]):

    text: Optional[str] = None

    def commit(self, db_session: Session):
        
        if not self.text:
            raise ValueError("Minimal requirements for commits are not respected : text")

        res = db_session.merge(
            Content(
                id = self.id,
                text = self.text,
            )
        )

        db_session.flush()
        self.id = res.id  # Make sure the id is synchronized with the database
    
    @classmethod
    def from_context(cls, db_session: Session, context: Context) -> List[Self]:
        """Returns the result of from_set using the content set returned by context.get_contents()"""
        return cls.from_set(
            context.get_contents(db_session)
        )
