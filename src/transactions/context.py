
from typing import (
    List,
    Dict,
    Union,
    Optional,
    Iterable,
)

from ..database import (
    Accessor, Content,
    NodeSet, 
    ACCESSORS, 
    LayerName,
)
from sqlalchemy.orm import Session


class AccessorContext(List[str]):

    def __init__(self, initial_value: Optional[Union[str, Iterable[str]]] = None):
        if isinstance(initial_value, str):
            initial_value = [initial_value]
        super().__init__(initial_value)
    
    def accessors(self, db_session: Session, layer: LayerName) -> NodeSet[Accessor]:

        AccessorORM = ACCESSORS[layer]

        result = NodeSet[Accessor]()

        for input in self:
            result += AccessorORM.vector_search(db_session, input, max_output=1)
            
        return result
        

class Context(Dict[LayerName, AccessorContext]):

    def __init__(self, **kwargs: Optional[Union[List[str], AccessorContext]]) -> None:
        super().__init__()
        for key, value in kwargs.items():
            self.__setitem__(key, value)

    def __setitem__(self, key: LayerName, value: Union[str, List[str], AccessorContext]) -> None:
        if not isinstance(value, AccessorContext):
            value = AccessorContext(value)
        super().__setitem__(key, value)

    def accessors(self, db_session: Session) -> Dict[LayerName, NodeSet[Accessor]]:

        return {
            layer_name : accessor_context.accessors(db_session)
            for layer_name, accessor_context in self.items()
        }

    def contents(self, db_session: Session) -> NodeSet[Content]:

        accessors = NodeSet[Accessor]()
        
        for layer_name, accessor_context in self.items():
            accessors += accessor_context.accessors(db_session, layer=layer_name)

        contents = NodeSet[Content]()

        for accessor in accessors:
            contents += accessor.next_contents(db_session)

        return contents
