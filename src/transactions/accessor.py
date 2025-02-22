
from typing import (
    List,
    Dict,
    Literal,
    Optional,
    Union,
)
from .node_transaction import NodeTransaction
from sqlalchemy.orm import Session
from ..database import Content, Accessor, Edge, NodeSet, LayerName, ACCESSORS
from ..database.config import WEIGHT_STRONG_INCREASE, WEIGHT_INCREASE, WEIGHT_DECREASE, WEIGHT_STRONG_DECREASE, EDGE_DELETE_CONDITION

from .content import ContentTransaction


class AccessorTransaction(NodeTransaction[Accessor]):

    proxy: Optional[int] = None
    embeddable: Optional[str] = None

    layer_name: LayerName

    @property
    def AccessorORM(self) -> Accessor:
        return ACCESSORS[self.layer_name]
    
    def proxies(self, db_session: Session) -> List['AccessorTransaction']:
        
        if self.id is None and self.proxy is None:
            raise ValueError('id or proxy id must be specified when looking for proxies.')
        
        if self.proxy is not None:
            target_id = self.proxy

        else:
            target_id = self.id

        proxies = db_session.query(self.AccessorORM).filter(
            self.AccessorORM.id == target_id 
            or self.AccessorORM.proxy == target_id
        ).all()

        return self.from_set(proxies)


    def commit(self, db_session: Session):

        if self.embeddable is None:
            raise ValueError("Minimal requirements for commits are not respected: embeddable")

        accessor_obj: Accessor = self.AccessorORM(
            id=self.id,
            embeddable=self.embeddable,
            proxy=self.proxy
        )
        accessor_obj.update_vector()
        
        res = db_session.merge(accessor_obj)
        
        db_session.flush()
        self.id = res.id  # Make sure the id is synchronized with the database


    def contents(self, db_session: Session) -> NodeSet[Content]:
        """
        Compute and return a NodeSet of Content objects from the accessor's contents.
        """

        accessor_instance: Accessor = db_session.query(self.AccessorORM).get(self.id)

        if accessor_instance is None:
            raise ValueError("Accessor not found.")

        return accessor_instance.get_contents(db_session)


    def set_edge(
            self, 
            db_session: Session, 
            content_tx: ContentTransaction, 
            weight: Optional[Union[float, Literal["decrease", "increase", "strong decrease", "strong increase"]]] = None
        ):
        """
        Update or create an edge between this accessor and the content.
        
        Parameters:
            - content_tx: The ContentTransaction the edge will point to.
            - weight: Either a float representing the new weight or a weight update literal
                    ("strong increase", "small increase", "strong decrease", "small decrease").
            
        If a literal is provided, then if an edge exists its weight is updated using the corresponding
        function; if not, a new edge is created with the default weight updated using the corresponding function.
        """

        if content_tx.id is None:
            raise ValueError("Must provide a content that has a valid id. If it's a new content run commit.")
        if self.id is None:
            raise ValueError("The accessor does not have an id. Commit it first to generate a new id")

        EdgeORM = self.AccessorORM.EdgeORM

        edge = db_session.query(EdgeORM).filter(
            EdgeORM.accessor_id == self.id,
            EdgeORM.content_id == content_tx.id
        ).first()

        if edge is None:
            edge = EdgeORM(accessor_id = self.id, content_id = content_tx.id)
            db_session.add(edge)
            db_session.flush()

        # Update the weight
        if weight:
            if isinstance(weight, str):
                match weight:
                    case "decrease":
                        weight = WEIGHT_DECREASE(edge.weight)
                    case "increase":
                        weight = WEIGHT_INCREASE(edge.weight)
                    case "strong decrease":
                        weight = WEIGHT_STRONG_DECREASE(edge.weight)
                    case "strong increase":
                        weight = WEIGHT_STRONG_INCREASE(edge.weight)
                    case _:
                        raise ValueError(f"Invalid value \"{weight}\" for the weight argument.")
            
            if EDGE_DELETE_CONDITION(weight):
                db_session.delete(edge)
            else:
                edge.weight = weight
                db_session.merge(edge)
