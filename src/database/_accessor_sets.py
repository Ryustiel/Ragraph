"""
Context graphs represent structured information, organized by contexts in various layers.

On single "content" layer is linked and accessed by multiple "accessor" layers, via weighted links.
Contents can be linked to one another when they represent very similar but very dense information.
Contents also come with some metadata that may impact their "relevancy" score and make them come up more frequently. (for example, whether a content is an instruction or not.)

Each layer has its own family of accessor nodes which represent a different "characterization" of context.
For example, a context layer can represent a "who the participant in the conversation are" type of context.

Together, the many layers of accessor and their varying connection to contents represent relevancy of information.

[A complex set of Graph-Vector Store layers]
"""

from typing import List, Dict, Type

from .contents import Content, Accessor
from ._accessor_config import LayerData
from sqlalchemy.orm import Session
from sqlalchemy import text

class WeightedContent:
    def __init__(self, content: Content, weight: float):
        self.content = content
        self.weight = weight

    def accessor_weight_formula(self, link_strength: float) -> float:
        return link_strength * self.weight

    def compute_accessors(self, accessor_layer: Type[Accessor]) -> 'AccessorSet':
        accessors = AccessorSet(accessor_layer=accessor_layer)
        for link_strength, accessor in getattr(self.content, accessor_layer.name):  # TODO : Calculate the name of the table linked to the accessor layer identifier... Create something uniform
            new_weight = self.accessor_weight_formula(link_strength)
            accessors.force_add(accessor, new_weight)
        return accessors

class ContentSet:
    """
    Determines operations on a group of contents.
    This class contains the various operations that can affect one or many sets of content nodes (and their linked accessors).
    """
    contents: List[WeightedContent]

    def __init__(self):
        self.contents = []

    def force_add(self, content: Content, weight: float):
        """
        Adds the new content to the set WITHOUT checking if it already exists.
        Use if confident in that it cannot happen.
        """
        self.contents.append(WeightedContent(content, weight))

    def update(self, content: Content, weight: float):
        """
        Adds or updates a content element in this set,
        using max(current_weight, provided_weight) in case of an update.
        """
        for existing_content in self.contents:
            if existing_content.content == content:
                existing_content.weight = max(existing_content.weight, weight)
                return
        self.contents.append(WeightedContent(content, weight))

    def __sum__(self, other: 'ContentSet') -> 'ContentSet':
        """
        Summing two content sets results in a new set with the union of contents, 
        with new weights representing the combination of both.
        """
        for weighted_content in other.contents:
            self.update(weighted_content.content, weighted_content.weight)

    def compute_accessors(self, accessor_layer: Type[Accessor]) -> 'AccessorSet':
        return sum([content.compute_accessors(accessor_layer=accessor_layer) for content in self.contents])

class WeightedAccessor:
    def __init__(self, accessor: Accessor, weight: float):
        self.accessor = accessor
        self.weight = weight

    def content_weight_formula(self, link_strength: float) -> float:
        return self.weight * link_strength

    def compute_contents(self) -> ContentSet:
        """
        Compute the node's contents and their weights.
        """
        content_set = ContentSet()
        for link_strength, content in self.accessor.contents:
            new_weight = self.content_weight_formula(link_strength)
            content_set.force_add(content, new_weight)
        return content_set

class AccessorSet:
    """
    Determines operations on a group of accessor nodes.
    This class contains the various operations that can affect one or many sets of accessor nodes.
    """
    accessors: List[WeightedAccessor]
    accessor_layer: Type[Accessor]

    def __init__(self, accessor_layer: Type[Accessor]):
        self.accessors = []
        self.accessor_layer = accessor_layer

    def force_add(self, accessor: Accessor, weight: float):
        """
        Adds the accessor without checking if it already exists.
        Use when confident that conflicts cannot happen.
        """
        self.accessors.append(WeightedAccessor(accessor=accessor, weight=weight))

    def update(self, accessor: Accessor, weight: float):
        """
        Add or update an accessor element to this set
        Using max(current_weight, provided_weight) if it already exists
        """
        for existing_accessor in self.accessors:
            if existing_accessor == existing_accessor.accessor:
                existing_accessor.weight = max(existing_accessor.weight, weight)
                return
        self.accessors.append(WeightedAccessor(accessor=accessor, weight=weight))

    def __sum__(self, other: 'AccessorSet') -> 'AccessorSet':
        """
        Summing two accessor sets results in a new set with the union of accessors, 
        with new weights representing the combination of both.
        """
        for other_accessor in other.accessors:
            self.update(other_accessor.accessor, other_accessor.weight)

    def next(self) -> 'AccessorSet':
        """
        Compute the accessor set containing the next accessors. 
        """
        # 1. Compute the contents set
        # 2. Compute the accessors sets from the contents set
        # 3. Create a new set that is the current combined to it via sum
        # 4. Return the sum
        next_contents = self.compute_contents()
        next_accessors = next_contents.compute_accessors()
        return self + next_accessors

    def compute_contents(self) -> ContentSet:
        """
        Compute the content nodes linked to this accessor AND their weights.
        """
        # NOTE : if a content is too frequent, the weight of its bonds with each accessor should diminish if it's not useful
        # NOTE : Need a method for evaluating efficiently if a chunk was used in a response or not (could be using embedding chunks in the response and comparing)
        # NOTE : Could be used in a writer's response as well.
        return sum([weighted_accessor.compute_contents() for weighted_accessor in self.accessors])  # TODO : Unwrap this operation so it doesnt take up as much memory

    @classmethod
    def from_vector(cls, session: Session, accessor_layer: Type[Accessor], layer_data: LayerData, vector: List[float]) -> 'AccessorSet':
        """
        - This extracts the closest accessor to the input vector.
        - What's called a DRIFT mechanism will cause the vector to shift slightly when receiving a match.
        (shift towards that match, so that it will tend towards the match)
        - If no close match is found, create a new context.
        - The context's label is the latest item that was used in retrieval if it matched the context more than a certain threshold.
        """
        accessor_set = AccessorSet(accessor_layer=accessor_layer)
        input_vector = f"[{', '.join(map(str, vector))}]"

        query = session.execute(text(
            f"""
            SELECT id, embedding
            FROM {accessor_layer.__tablename__}
            ORDER BY embedding <=> :input_vector
            LIMIT 4
            """), 
            {'input_vector': input_vector}
        )
        results = query.fetchall()

        for row in query.fetchall():
            print(row)

        if results:
            print("Found a result")
            # Apply drift
            # accessor_set.update(best_match)

        else:
            print("No result found")
            new_accessor = session.add(accessor_layer(embedding=vector))
            accessor_set.update(new_accessor)
            
        return accessor_set
    