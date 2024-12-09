"""
PairManager is the base object that interacts with the database.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Callable, Generator, Optional
from pydantic import BaseModel

class BaseNode: pass  # For type hints


# ================================================================= PLOT FORMAT MODELS ===================================================================


class NodePlot(BaseModel):
    id: str
    type: str
    chunk: str
    label: str

class EdgePlot(BaseModel):
    start: str
    end: str
    type: str
    label: str


# =================================================================  BASE PAIR GRAPH STRUCTURE  ===========================================================


class PairGraph(ABC):
    """
    Represents a pair of tables from a database or a graph collection.
    """
    def __init__(
            self, 
            pair_name: str, 
            embedding_function: Callable[[str], List[float]],
            content_merge_function: Optional[Callable[[str, str], str]] = None,
            accessor_merge_function: Optional[Callable[[str, str], str]] = None,
            content_similarity_threshold: float = 0.5,
            accessor_similarity_threshold: float = 0.5,
            content_equality_threshold: float = 1,
            accessor_equality_threshold: float = 1,
        ):
        self.name = pair_name
        self.changes: Dict[int, 'BaseNode'] = dict()

        self.embedding_function = embedding_function
        self.content_merge_function = content_merge_function
        self.accessors_merge_function = accessor_merge_function

        # Thresholds
        self.content_similarity_threshold = content_similarity_threshold
        self.accessor_similarity_threshold = accessor_similarity_threshold
        self.content_equality_threshold = content_equality_threshold
        self.accessor_equality_threshold = accessor_equality_threshold

    def __eq__(self, other: 'PairGraph') -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name
    
    def __hash__(self) -> int:
        return hash(self.name + self.__class__.__name__)

    # ---------------------------------------------------------------- COMMIT ----------------------------------------------------------------

    def register(self, node: 'BaseNode'):
        """
        Register or update the node so that its changes can be saved on commit.
        """
        h = hash(node)
        self.changes[h] = node

    def get(self, node: 'BaseNode'):
        """
        Look for the node in the changes registry.
        """
        h = hash(node)
        if h in self.changes.keys():
            return self.changes[h]
        else: 
            return None

    def commit(self):
        """
        Commit the nodes
        """
        deleted_keys = list()

        # 1. For each BUILT node, if already synced, do nothing.
        for node in self.changes.values():

            if node.is_synced:
                continue

            # 2. If not synced, delete the node from the collection if its node is deleted, remove it from the changes and delete it from the database.
            if node.is_deleted:
                node.database_delete()
                deleted_keys.append(hash(node))
                continue

            # 3. If not to be deleted and not synced, commit the changes to the database to merge the existing item
            node.database_update()

        # 4. Set synced to True. (but keep the node on the "changes" list)
        self.changes = {k: v for k, v in self.changes.items() if k not in deleted_keys}  # This prevents the node from being provided by the PairGraph object.

    # ---------------------------------------------------------------- CREATE ----------------------------------------------------------------

    @abstractmethod
    def Accessor(self, input: str, is_node_id: bool = False) -> 'BaseNode':
        """
        Create a new accessor node object from the provided data.
        """
        pass

    @abstractmethod
    def Content(self, input: str, is_node_id: bool = False) -> 'BaseNode':
        """
        Create a new content node object from the provided data.
        """
        pass

    # ---------------------------------------------------------------- PLOT --------------------------------------------------------------------

    @abstractmethod
    def get_nodes(self) -> Generator[NodePlot, None, None]:
        """
        Produce the list of nodes in the current graph.
        """
        pass

    @abstractmethod
    def get_edges(self) -> Generator[EdgePlot, None, None]:
        """
        Produce the list of edges in the current graph.
        """
        pass
