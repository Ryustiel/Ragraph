"""
Describe the main object the user will be interacting with,
to add and retrieve data from the graph.
"""
import numpy

from typing import (
    List, 
    Dict,
    Any,
    Union,
    Literal,
)

from sqlalchemy.orm import Session

from ._config import (
    CONTENT_SIMILARITY_CONDITION, 
    CONTENT_IDENTITY_CONDITION,
    CONTENT_EMBEDDINGS_FUNCTION,

    WEIGHT_DECREASE,
    WEIGHT_INCREASE,
    WEIGHT_STRONG_DECREASE,
    WEIGHT_STRONG_INCREASE,
    EDGE_DELETE_CONDITION,
)
from ._vectorizer import Vectorizer
from ._postgre import LatentChunk, Content, ContentSet, Accessor, ACCESSORS
from .naming import attribute_name



class WeightUpdate(Dict[int, Literal["strong decrease", "decrease", "hold", "increase", "strong increase"]]):
    """
    Represents a mapping of content_ids to weight update labels.
    """
    pass



class Context(Dict[str, Accessor]):
    """
    Represents a set of Accessors that can be used to get Content nodes.

    The context is a dict of shape {<layer name>: <accessor instance from the db>}
    """

    def insert_content(self, session: Session, text: str):
        """
        Inserts the new content into the database or updates the existing similar content node.
        Makes it accessible via the provided context if it's not already.

        NOTE : This method does not MODIFY existing content. It only adds 3 types of things : contents, edges and latent_chunks, 
        to be processed by other processes triggered by other events.
        NOTE : Updates to the weight are caused by the other mechanisms (retrieval evaluation | inquiry events)
        """

        input_embedding = CONTENT_EMBEDDINGS_FUNCTION(text)
        similar_contents = Content.find_content(session, input_embedding)
        
        selected_node = None
        if similar_contents:
            most_similar = similar_contents[0]

            if CONTENT_SIMILARITY_CONDITION(most_similar.embedding, input_embedding):
                # Select the node that is similar enough
                selected_node = most_similar
                if CONTENT_IDENTITY_CONDITION(most_similar.embedding, input_embedding):
                    # Store the current chunk as a latent chunk to be added later to the content node
                    session.add(LatentChunk(text=text, content_id=selected_node.id))
                else:
                    pass # Ignore the content text chunk as it's too similar to what's in the content node already

        if selected_node is None:  # Create a new content node
            selected_node = Content(text=text, embedding=input_embedding)
            session.add(selected_node)
            session.flush()  # We shouldnt be committing yet because the edges are missing, but we need to generate the new node id

        # Adds the accessors to the content.
        for layer_name, accessor in self.items():
            # Creates a new edge if one doesn't already exist
            if not accessor.edges.exists(session, content_id=selected_node.id, accessor_id=accessor.id):
                new_edge = accessor.edges(content_id=selected_node.id, accessor_id=accessor.id)
                session.add(new_edge)  # Do not update the existing edge if it exists already
            else:
                # Improve the weight of content links that already exist as the attempt to add it symbolizes that it was important to know here
                existing_edge = accessor.edges.get_edge(session, content_id=selected_node.id, accessor_id=accessor.id)
                if existing_edge:
                    existing_edge.weight = WEIGHT_STRONG_INCREASE(existing_edge.weight)

        session.commit()
        
    
    def get_content(self, session: Session) -> ContentSet:
        """
        Compute the direct (first degree) contents that can be accessed from this context.
        """
        result = ContentSet()
        for content_set in [accessor.get_contents(session) for accessor in self.values()]:
            result += content_set
        return result


    def update_weights(self, session: Session, weight_update: WeightUpdate):
        """
        1. For each accessor in the current context, get the edges associated to the content nodes referenced as keys in weight_update
        2. For each edge, apply the corresponding weight update formula.
        NOTE : Multiple edges typically connect to a single content node, and will each be applied the same formula.
        3. If the DELETE_CONDITION is valid for the new weight on an edge, it is deleted.
        """
        for layer_name, accessor in self.items():
            for content_id, update_label in weight_update.items():
                edge = accessor.edges.get_edge(session, content_id=content_id, accessor_id=accessor.id)
                if edge:
                    match update_label:
                        case "strong decrease":
                            edge.weight = WEIGHT_STRONG_DECREASE(edge.weight)
                            if EDGE_DELETE_CONDITION(edge.weight):
                                session.delete(edge)
                        case "decrease":
                            edge.weight = WEIGHT_DECREASE(edge.weight)
                            if EDGE_DELETE_CONDITION(edge.weight):
                                session.delete(edge)
                        case "hold":
                            pass
                        case "increase":
                            edge.weight = WEIGHT_INCREASE(edge.weight)
                        case "strong increase":
                            edge.weight = WEIGHT_STRONG_INCREASE(edge.weight)
                        case _:
                            raise ValueError(f"Invalid weight update label: {update_label}")
        session.commit()


    @classmethod
    def from_input(cls, session: Session, input: Dict[str, Union[str, int]]) -> 'Context':
        """
        Produce a context dict from a context input of shape {<layer name>: <input string>}
        A vector search will be performed to find or create a relavant accessor node from each requested layer.
        """
        compiled_context = cls()
        for layer_name, context_input in input.items():
            
            if layer_name not in ACCESSORS.keys():
                raise ValueError(f"Layer {layer_name} provided in the context does not exist in the database. (Available layers: {ACCESSORS.keys()}")    

            if isinstance(context_input, str):
                compiled_context[layer_name] = ACCESSORS[layer_name].from_text(session, text=context_input)
            elif isinstance(context_input, int):
                node = ACCESSORS[layer_name].from_id(session, id=context_input)
                if node is None:
                    raise ValueError(f"No node found for ID {context_input} in layer {layer_name}.")
                compiled_context[layer_name] = node
            else:
                raise ValueError(f"Invalid input type for layer {layer_name}. Expected str or int, got {type(context_input).__name__}")

        session.commit()  # Commits the new, added accessors if any

        return compiled_context
