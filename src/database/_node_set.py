
from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    TypeVar,
    Generic,
    Callable,
    Iterator,
)
from ._A_Edge_Node import Node

T = TypeVar("NodeLike", bound=Node)


class NodeSet(Dict[T, Tuple[float, int]], Generic[T]):
    """
    Dict is in format : 
    "Node, (weight, count)"
    """

    def __init__(self, initial_value: Optional['NodeSet[T]'] = None):
        
        if initial_value:
            self.update(initial_value)

        # This value will be computed anew everytime the set is modified.
        # It will be used in combination with the "count" value in the "count_weight()" method.
        self.__count_divider = None

    @property
    def count_divider(self):
        if self.__count_divider is None:
            self.__count_divider = max([val[1] for val in self.values()])
        return self.__count_divider

    def __add__(self, other: 'NodeSet[T]') -> 'NodeSet[T]':
        """
        Summing two node sets results in a new set with the union of the sets, 
        * count is increased by one if the node was present in the two sets, 
        * weight is the max value of the weights of the two sets. 
        """
        # Pre iterating on keys to avoid interacting with them during the loop below which alters them
        # XXX : I'm not even sure that .keys() is an iterator, might be a list already so...
        existing_nodes = list(self.keys())

        for other_key in other.keys():
            if other_key in existing_nodes:
                # If the weight is to be updated, then the count should be too
                (weight_a, count_a), (weight_b, count_b) = self[other_key], other[other_key]
                new_weight, new_count = max(weight_a, weight_b), sum((count_a, count_b))
            else:
                (new_weight, new_count) = other[other_key]

            self[other_key] = (new_weight, new_count)

        # Marks the count divider for recalculation
        self.__count_divider = None

        return self

    def __mul__(self, factor: float) -> 'NodeSet[T]':
        for key, weight in self.items():
            self[key] = weight * factor
        return self
    
    def trim(self, weight: Optional[float] = None, count: Optional[int] = None):
        """
        Remove all nodes that have weight or count values below the specified threshold.
        """

        for key in list(self.keys()):
            node_weight, node_count = self[key]
            if weight:
                if node_weight < weight:
                    del self[key]
                    continue
            if count:
                if node_count < count:
                    del self[key]
                    continue
    
    def get_nodes_weighted(self) -> List[T]:
        """Returns the nodes sorted by weight descending."""
        return [
            node for node, _ in sorted(
                list(self.items()), 
                key = lambda key_val: key_val[1][0], # Sorting by the weight 
                reverse = True,
            )
        ]

    def get_nodes_custom(
            self, 
            score_function: Callable[[Tuple[float, float]], float]
        ) -> List[T]:
        """
        Sort the node by score descending,
        according to the custom score_function.
        
        Parameters:
            * score_function : "(weight: float, count_score: float) -> score: float", Compute a unique score for sorting each node
        """
        scores: List[float] = [
            score_function(
                weight, count / self.count_divider
            )
            for node, (weight, count) in self.items()
        ]

        return [
            node for node, _ in sorted(
                zip(self.keys(), scores),
                key = lambda x: x[1],  # Sorting by score
                reverse = True,
            )            
        ]
    
    def get_nodes_counted(self) -> List[T]:
        """Returns the nodes sorted by count descending."""
        return self.get_nodes_custom(
            score_function = lambda weight, count_score: count_score
        )
