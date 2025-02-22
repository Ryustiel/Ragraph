
from typing import (
    List,
    Self,
    Optional,
    TypeVar,
    Generic,
    Union,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..database import Node, NodeSet


T = TypeVar('NodeLike', bound=Node)


class NodeTransaction(BaseModel, Generic[T]):
    
    model_config = {"from_attributes": True}

    id: Optional[int] = None

    @classmethod
    def from_set(cls, node_set: Union[NodeSet[T], List[T]]) -> List[Self]:
        transactions: List[Self] = []
        for node in node_set:
            transactions.append(
                cls.model_validate(node)
            )
        return transactions
