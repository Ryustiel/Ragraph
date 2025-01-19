"""
Define the interfacing models, that accept outside information or produce exportable models.
Most of these models must be pydantic because they will be interacted with by the FastAPI layer.
"""
from typing import (
    Dict,
    List,
)
from sqlalchemy.orm import Session

from .context import (
    Content, 
    Accessor, 
    ContentSet, 
    AccessorSet, 
    ACCESSORS, 
    LAYER_DATA,
    BaseContextInput, 
    EmbeddingsVector,
    table_name_from_label,
)


class ContextInput(BaseContextInput):
    """
    Lets you define the current context and extract content nodes from the various accessor layers based on it.
    """

    def compute_vectors(self) -> Dict[str, EmbeddingsVector]:
        """
        Compute the vectors for each accessor from the input
        """
        vectors = {}

        for layer_name, data in self.context.items():
            if layer_name not in ACCESSORS.keys():
                pass  # Skipping the parsing step for any layer that was not registered in context.accessors.py
            else:
                layer = LAYER_DATA[layer_name]

                if not isinstance(data, layer.get_input_type()):
                    raise ValueError(f"Layer type mismatch or unregistered data type: The layer is typed {layer.get_input_type()} while the data is typed {type(data)}.")
                
                else:
                    vectors[layer_name] = layer.embeddings_function(data)

        return vectors    

    def compute_accessors(self, session: Session) -> Dict[str, AccessorSet]:
        """
        Extract a set of accessors from the input.
        First calculate vectors for each accessor then run a match for each layer.
        The result is a Dict[<layer name>, <set of 1 single accessor>]        
        """
        layer_data = self.compute_vectors()

        for layer_name, vector in layer_data.items():
            layer_data[layer_name] = AccessorSet.from_vector(
                session=session, 
                accessor_layer=ACCESSORS[layer_name],
                layer_data=LAYER_DATA[layer_name], 
                vector=vector,
            )

        return layer_data

    def compute_contents(self, session: Session, max_depth: int) -> ContentSet:
        """
        Extract a set of weighted contents from the current context.
        """
        # 1. Use the accessors from the accessor set to get the contents.
        # 2. Without summing the content sets, get the next generation of accessors from the contents for each layer.
        # 3. Sum the weighted contents for all 2 generation of weighted contents. Repeat step 2+3 as many times as "pre recursion" is needed, and for each layer.
        # 4. After going to some depth, merge the contents of all layers.
        
        # Compute next accessors for each layer
        accessors = self.compute_accessors(session)
        for _ in range(max_depth):
            for layer_name in accessors.keys():
                accessors[layer_name] = accessors[layer_name].next()
        
        # Compute contents from all of the layers
        contents = ContentSet()
        for layer_name, accessor_set in accessors.items():
            contents.add(accessor_set.compute_contents())

        return contents

    def insert_content(self, session: Session, chunks: List[str], merge_threshold: float, identity_threshold: float):
        """
        Insert the contents with the current accessors.
        """
        if merge_threshold > identity_threshold:
            raise ValueError("merge_threshold must be less than identity_threshold")
        
        # 1. Comupute accessors based on the current context
        # 2. Look through all the contents for any similar enough chunk, if there's one similar enough, do a merge.
        # If extremely similar do nothing, but start from the existing content.
        # 3. Connects the contents to the accessors or strengthen the bond.
        accessors = self.compute_accessors(session)
        contents = [
            Content(
                text=chunk,
                vector=""
            )
            for chunk in chunks
        ]

        for accessor_set in accessors.values():
            for accessor in accessor_set.iterate_accessors():
                for content in contents:
                    # Add a link to content
                    # TODO : Update accessor items instead of content items when it comes to linking (so that we can avoid cross linking)
                    pass

        for content in contents:
            session.merge(content)

    def update_weights_from_review(self, review: Dict[int, bool], min_weight: float, decay_rate: float):
        """
        Update the weights of the links between the contextual accessors
        and the target content according to the review.

        review is a dict : keys are ContentID, values are bool indicating whether the content was used in the LLM's response or not.
        if a key was used, weight is increased towards 1.0 using a ... function.

        new_weight = max(min_weight, f(accessor_weight, review, decay_rate))

        min_weight is the minimum weight a link between two nodes can have after this update is ran.
        decay_rate is a factor that determines how fast a link should decay depending on the rating.
        """
        # 1. For each weighted accessor in each layer, update the weight as a 
        pass
