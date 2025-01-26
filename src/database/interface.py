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
)

from sqlalchemy.orm import Session

from ._config import (
    CONTENT_SIMILARITY_CONDITION, 
    CONTENT_IDENTITY_CONDITION,
    CONTENT_EMBEDDINGS_FUNCTION,
)
from ._vectorizer import Vectorizer
from ._postgre import LatentChunk, Content, ContentSet, Accessor, ACCESSORS
from .naming import attribute_name


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
            session.flush()  # We shouldnt be committing yet because the edges are missing

        # Adds the accessors to the content.
        for layer_name, accessor in self.items():
            # Creates a new edge if one doesn't already exist
            if not accessor.edges.exists(session, content_id=selected_node.id, accessor_id=accessor.id):
                new_edge = accessor.edges(content_id=selected_node.id, accessor_id=accessor.id)
                session.add(new_edge)  # Do not update the existing edge if it exists already

        session.commit()
        
    
    def get_content(self, session: Session) -> ContentSet:
        """
        Compute the direct (first degree) contents that can be accessed from this context.
        """
        result = ContentSet()
        for content_set in [accessor.get_contents(session) for accessor in self.values()]:
            result += content_set
        return result

    @classmethod
    def from_input(cls, session: Session, input: Dict[str, str]) -> 'Context':
        """
        Produce a context dict from a context input of shape {<layer name>: <input string>}
        A vector search will be performed to find or create a relavant accessor node from each requested layer.
        """
        compiled_context = cls()
        for layer_name, context_input in input.items():
            
            if layer_name not in ACCESSORS.keys():
                raise ValueError(f"Layer {layer_name} provided in the context does not exist in the database. (Available layers: {ACCESSORS.keys()}")    

            compiled_context[layer_name] = ACCESSORS[layer_name].from_text(session, text=context_input)

        session.commit()  # Commits the new, added accessors if any

        return compiled_context
