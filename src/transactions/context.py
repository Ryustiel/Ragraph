
from typing import (
    List,
    Dict,
)

from ..database import (
    Accessor, Content,
    NodeSet, 
    ACCESSORS, 
    LayerName,
)
from sqlalchemy.orm import Session



class AccessorContext(List[str]):
    
    
    def get_accessors(self, db_session: Session, layer: LayerName) -> NodeSet[Accessor]:

        AccessorORM = ACCESSORS[layer]

        result = NodeSet[Accessor]()

        for input in self:
            result += AccessorORM.vector_search(db_session, input, max_output=1)
            
        return result
        


class Context(Dict[str, AccessorContext]):
    

    def get_accessors(self, db_session: Session) -> Dict[LayerName, NodeSet[Accessor]]:

        return {
            layer_name : accessor_context.get_accessors(db_session)
            for layer_name, accessor_context in self.items()
        }
    

    def get_contents(self, db_session: Session) -> NodeSet[Content]:

        accessors = NodeSet[Accessor]()
        
        for layer_name, accessor_context in self.items():
            accessors += accessor_context.get_accessors(db_session, layer=layer_name)

        contents = NodeSet[Content]()

        for accessor in accessors:
            contents += accessor.get_contents(db_session)

        return contents
