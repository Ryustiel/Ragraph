
"""
Nodes expose custom operations 
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set, Generator, Tuple, Optional

from .pairs import PairGraph

DEBUG_BUILD = False
DEBUG_OPS = False


# ================================================================= BASE NODE =================================================================


class BaseNode(ABC):
    """
    Represents a node in the context conn graph.
    """

    def __init__(
        self,
        pair: 'PairGraph',
        chunk: str = None,
        node_id: str = None,
        vector: List[float] = None,
        is_synced: bool = False,
        is_built: bool = False,
    ):
        if chunk is None and node_id is None:
            raise ValueError("Either chunk or node_id must be provided to build a node.")
        
        if not is_synced and chunk is not None and node_id is not None:
            raise ValueError("Cannot provide both chunk and node_id without explicitely setting is_synced=True. (Search mode = False)")

        self.pair = pair
        self._chunk = chunk
        self._node_id = node_id
        self._vector = vector

        # True if the node is in sync with the database
        self.is_synced = is_synced
        self.is_built = is_built
        self.is_deleted = False

    @abstractmethod
    def __eq__(self, other: 'BaseNode') -> bool:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass

    # ---------------------------------------------------------------- PROPERTY METHODS ----------------------------------------------------------------

    @property
    def node_id(self) -> str:
        if self._node_id is None:
            if DEBUG_OPS: print("NODE ID RETRIEVAL TRIGGERED BUILD FOR", self._node_id)
            if self._chunk is None:
                raise ValueError("Either chunk or node_id must be provided to build a node.")
            self.build()
        return self._node_id

    @node_id.setter
    def node_id(self, value: str):
        self._node_id = value
        self.is_synced = False

    @property
    def chunk(self) -> str:
        if self._chunk is None:
            if DEBUG_OPS: print("CHUNK RETRIEVAL TRIGGERED BUILD FOR", self._node_id)
            if self._node_id is None:
                raise ValueError("Either chunk or node_id must be provided to build a node.")
            self.build()
        return self._chunk

    @chunk.setter
    def chunk(self, value: str):
        self._chunk = value
        self.is_synced = False

    @property
    def vector(self) -> List[float]:
        if self._vector is None:
            self._vector = self.pair.embedding_function(self._chunk)  # Callable[[str], List[float]]
            self.is_synced = False
        return self._vector
    
    # ---------------------------------------------------------------- BUILD AND COMMIT ----------------------------------------------------------------

    def build(self):
        if DEBUG_BUILD: print("\n>> BUILDING", self._node_id)

        # 1. If already built, do nothing.
        if self.is_built:
            return

        elif self.is_deleted:
            raise ValueError("The current node has been deleted and thus cannot be built.")
        
        # 2. Look for the node in the pair changes registry (=whether it has already been loaded). Merge it with the current node if it exists.
        found_match = False
        if self._node_id is not None:
            for other in self.pair.changes.values():
                if (
                    other._node_id is not None 
                    and self._node_id == other._node_id
                    and isinstance(other, self.__class__) # ID and NODE TYPE should be the same
                    and other.is_built  # Only use the other node if it's already built (otherwise simply replace it)
                    # NOTE : When nodes are not built, they are considered read only. Any operation that modifies a node will trigger a build.
                ):

                    if DEBUG_BUILD: print("MERGING WITH PAIR REGISTRY")

                    self._merge(other)
                    found_match = True

        if not found_match:

            # 3. Look for the node in the database by id or by chunk.
            if self._from_id():  # True if the node has been updated using the node_id attribute.
                if DEBUG_BUILD: print("MERGED FROM ID")
                self.pair.register(self)  # Updates the node in the set.

            else:  # Looking for similar chunks
                other = self.query_equal()

                if other:
                    if DEBUG_BUILD: print("MERGED FROM RESULT")
                    self._merge(other)
                    self.pair.register(self)
                
                # 4. If it doesn't exist, create it using the provided information or default values.
                else:  # Node does not exist in the database as node_id or chunk : create new values
                    if DEBUG_BUILD: print("NEW")
                    self._as_new()
                    self.pair.register(self)
        
        # 7. Mark the node as built and add it to the local list of built nodes.
        self.is_built = True  # Node is now built
        self.pair.register(self)

    # ---------------------------------------------------------------- UTILITY -----------------------------------------------------------------

    @abstractmethod
    def chunk_merge(self, chunk: str) -> str:
        """
        Returns a new merged chunk generated from the current self.chunk and the new provided chunk.
        """
        return self.chunk + " " + chunk

    # ---------------------------------------------------------------- ABSTRACT ----------------------------------------------------------------

    @abstractmethod
    def query_equal(self) -> Optional['BaseNode']:
        """
        Find the node that is the most similar to the current node, if any.
        """
        pass

    @abstractmethod
    def query(self, max_results: int = 1, similarity_threshold: float = None) -> Generator['BaseNode', None, None]:
        """
        Query "max_results" nodes whose content is similar to this one, by similarity.
        """
        pass

    @abstractmethod
    def _merge(self, other: 'BaseNode'):
        """
        Merge this node with another one.
        """
        pass

    @abstractmethod
    def _from_id(self) -> bool:
        """
        Update itself by fetching a node in the database using the node_id.

        Returns:
            bool: Whether the update was successful (or could be initiated at all).
        """
        pass

    @abstractmethod
    def _as_new(self):
        """
        Initialize a brand new state of the object (locally).
        """
        if self._chunk is None:
            raise ValueError("Cannot create an AccessorNode without a chunk.")

        self._node_id = "0"  # Define here the correct id
        self._chunk = self._chunk
        self._contents = list()
        self._neighbors = dict()
        self.is_synced = False

    @abstractmethod
    def database_update(self):
        """
        Update itself in the database.
        """
        pass

    @abstractmethod
    def database_delete(self):
        """
        Delete the node for itself from the database.
        """
        pass

    @abstractmethod
    def delete(self):
        """
        Safely delete itself from the graph.
        """
        pass




# ================================================================= ACCESSOR =================================================================




class AccessorNode(BaseNode):
    """
    Represents an accessor node in the context graph.
    """
    def __init__(
        self,
        pair: 'PairGraph',
        chunk: str = None,
        node_id: int = None,
        contents: List['ContentNode'] = None,
        neighbors: Dict['AccessorNode', int] = None,
        vector: List[float] = None,
        is_synced: bool = False,
        is_built: bool = False,
    ):
        super().__init__(
            pair=pair,
            chunk=chunk,
            node_id=node_id,
            vector=vector,
            is_synced=is_synced,
            is_built=is_built,
        )
        self._contents = contents
        self._neighbors = neighbors

    def __eq__(self, other: 'BaseNode') -> bool:
        if not isinstance(other, AccessorNode):
            return False
        return self.node_id == other.node_id
    
    def __hash__(self) -> int:
        return hash('A' + str(hash(self.pair)) + self.node_id)

    # ---------------------------------------------------------------- PROPERTY METHODS ----------------------------------------------------------------

    @property
    def contents(self) -> List['ContentNode']:
        if not self.is_built:
            if DEBUG_OPS: print("CONTENT RETRIEVAL TRIGGERED BUILD FOR", self._node_id)
            self.build()
        return self._contents

    @property
    def neighbors(self) -> Dict['AccessorNode', int]:
        if not self.is_built:
            if DEBUG_OPS: print("NEIGHBOR RETRIEVAL TRIGGERED BUILD FOR", self._node_id)
            self.build()
        return self._neighbors
    
    # ---------------------------------------------------------------- DATA EDITION ----------------------------------------------------------------

    def link(self, content: 'ContentNode') -> 'ContentNode':
        """
        Add a new content node and update the neighbors.
        """
        # 1. Make sure the node has no duplicates.
        if content.node_id in self.contents:  # Content has already been added to this node.
            return content
        
        # 2. Add the content to the accessor node itself.
        else:
            self.contents.append(content)

        # 3. Neighbor update
        # Update the neighbors of the current accessor and the next accessors of the content node.
        for accessor in content.accessors:
            if accessor.node_id != self.node_id:

                # Add +1 to the weight (shared content count) to the other's neighbor list.
                if self.node_id not in accessor.neighbors.keys():
                    accessor.neighbors[self] = 1
                else:
                    accessor.neighbors[self] += 1

                # Add +1 to the weight to our own list of neighbors for this neighbor.
                if accessor.node_id not in self.neighbors.keys():
                    self.neighbors[accessor] = 1
                else:
                    self.neighbors[accessor] += 1

    def unlink(self, content: 'ContentNode') -> 'AccessorNode':
        """
        Disconnect the content node from the current accessor.
        """
        # 1. Remove the content node from the contents list (edges) of the current accessor
        found = False
        for i, linked_content in enumerate(self.contents):
            if linked_content == content:
                found = True
                self.contents.pop(i)
        
        if not found:
            return self
            raise ValueError(f"Tried to unlink a content node {content.node_id} from accessor {self.node_id} but it was not linked.")

        # 2. If the content node lost its last accessor, remove it from the graph. (No isolated content node)
        else:
            if len(content.accessors) <= 1:
                content.delete()

            # 3. Remove the neighboring weight from every node which was connected through this content node.
            # This is the reverse of the .link() method.
            for accessor in content.accessors:
                if accessor.node_id != self.node_id:

                    # Remove 1 from the weight of the other's neighbor list.
                    if accessor.neighbors[self] <= 1:  # Removing 1 will make the weight 0, so we can remove the node from the list.
                        accessor.neighbors.pop(self)
                    else:
                        accessor.neighbors[self] -= 1

                    # Remove 1 from the weight of our own list of neighbors for this neighbor.
                    if accessor.neighbors[self] <= 1:
                        self.neighbors.pop(accessor)
                    else:
                        self.neighbors[accessor] -= 1

    def delete(self):
        # 1. Unlink all content nodes
        for content in self.contents:
            self.unlink(content)

        # 2. Delete the node data to prevent accessing it, except the node_id so that it can still be tracked and deleted upon commit.
        self._chunk = None
        self._vector = None
        self._contents = None
        self._neighbors = None

        self.is_synced = True  # Prevents pushing new data from deleted nodes
        self.is_deleted = True

        if self._node_id is None:
            raise ValueError("Cannot delete a node without a node_id set. This should never happen.")

    # ---------------------------------------------------------------- DATA RETRIEVAL ----------------------------------------------------------------

    # CAN BE OVERRIDDEN
    @classmethod
    def process_children(cls, children: List['ContentNode']) -> List['ContentNode']:
        return children

    def get_content(self, max_results: int = 1, similarity_threshold: float = 0.5) -> Generator['ContentNode', None, None]:
        
        # Step 1:
        # Query for the root accessor nodes and initialize them with weight=1 (max).

        accessors: List[Tuple['AccessorNode', float]] = [(acc, 1) for acc in self.query(max_results, similarity_threshold)]

        # Step 2:
        # Store the content ids that have already been used, so that we don't use them again.

        used_up_content: Set[int] = set()
        used_up_accessors: Set[int] = set(acc.node_id for (acc, _) in accessors)

        while len(accessors) > 0:

            # Step 3:
            # Rerank the children of the top weight accessor and yield them one by one.

            (acc, weight) = accessors.pop(0)

            for child in acc.contents:  # NOTE : Use __class__.process_children here if needed.
                if child.node_id not in used_up_content:
                    yield child
                    used_up_content.add(child.node_id)

            n_content = len(acc.contents)  # Number of connected content nodes for computing the weights
            for neighbor, n_shared_content in acc.neighbors:

                if neighbor.node_id in used_up_accessors:  # Don't process the same node twice, prevents loops.
                    continue

                used_up_accessors.add(neighbor.node_id)  # Prevents that node from being processed twice.

                # Step 4:
                # Compute the new weight of the accessor's neighbors

                # NOTE : Neighbor new weight = ratio of shared content nodes (0<.<1)) * weight of source node
                neighbor_weight = weight * n_shared_content / n_content

                # Step 5:
                # Insert the new weighted node in the list and preserve the order (weight ascending).

                inserted = False
                for i, (_, w) in enumerate(accessors):
                    if neighbor_weight > w:
                        accessors.insert(i, (neighbor.node_id, neighbor_weight))
                        inserted = True
                        break
                if not inserted:
                    accessors.append((neighbor.node_id, neighbor_weight))

    # ---------------------------------------------------------------- UTILITY -----------------------------------------------------------------

    def chunk_merge(self, chunk: str) -> str:
        """
        Returns a new merged chunk generated from the current self.chunk and the new provided chunk.
        """
        return self.pair.accessors_merge_function(self.chunk, chunk)

    # ---------------------------------------------------------------- COMMIT ----------------------------------------------------------------

    def _merge(self, other: 'AccessorNode'):

        self.is_synced = other.is_synced  # Disabled on editing operations

        # 0. Check if the current node is synced (if not, the merge operation will be simplified)
        if self._node_id:  # If both nodes already exist and potentially have metadata, merge them. 
            # NOTE This should be a rare case.

            # 0. Initialize all of the other attributes as empty attributes if they don't exist already.

            if self._chunk is not None:

                # 1. Merge the chunks
                self._chunk = self.chunk_merge(other.chunk)

                # 2. Rebuild the vector from the new chunk
                self._vector = self.pair.embedding_function(self._chunk)

                self.is_synced = False

            else:
                self._chunk = other.chunk
                self._vector = other._vector  # Using _ prevents generating the same vector twice, or generating one that would never be used because the node is an existing readonly.

            self._node_id = other.node_id

            if self._neighbors is None and self._contents is None:
                # just use the other attributes
                self._contents = other.contents
                self._neighbors = other.neighbors
                if DEBUG_BUILD: print("MERGE ACTION = JUST RETRIVED EXISTING NODE FROM ID")

            elif self._neighbors is not None and self._contents is not None:
                # Merge the contents and neighbors

                # 3. Extract and merge the differences in contents and neighbors with the other node.
                shared_contents = list()
                for content in self._contents:
                    if content in other.contents:
                        shared_contents.append(content)

                # 4. Unlink the shared contents from the current node (to prevent duplicates)
                for content in shared_contents:
                    self.unlink(content)

                # 5. Copy the values from the other node into its own contents and neighbors. (to save the linking operation)
                # NOTE : The resulting node has a combination of the contents (and thus neighbor patterns) of the other node.

                # Copy contents
                self._contents.extend(other.contents)

                # Copy neighbors
                for neighbor, weight in other.neighbors.items():
                    if neighbor in self._neighbors.keys():
                        self._neighbors[neighbor] += weight
                    else:
                        self._neighbors[neighbor] = weight

                self.is_synced = False

                # NOTE : Steps 3-4-5-6 are meant to take advantage of the fact that the other node usually have more metadata 
                # because it already existed in the database before it was being merged, whereas self is likely a brand new node with fewer contents to unlink.

            else:
                raise ValueError(f"Neighbors {self._neighbors} and contents {self._contents} should be updated via the link methods.")

        else:  # Simplified merge operation : merge the chunks and replace all other fields of the current node with the other.
            # NOTE : This should be the most common case.

            # 1. Merge the chunks
            # Only the other chunk exists, simply use that chunk as self chunk.
            if self._chunk is None:
                # Bypass the self.chunk accessor to avoid calling the build() method
                self._chunk = other.chunk
            
            else:  # Both chunks exist : merging
                self._chunk = self.chunk_merge(other.chunk)
                if DEBUG_BUILD: print("MERGE ACTION = JUST RETRIEVED EXISTING NODE FROM CHUNK")
                self.is_synced = False

            # 2. Rebuild the vector from the new chunk
            self._vector = self.pair.embedding_function(self._chunk)

            # 3. Copy all of the values from the other node into the current node
            self._node_id = other.node_id
            self._contents = other.contents
            self._neighbors = other.neighbors




# ================================================================= CONTENT =================================================================




class ContentNode(BaseNode):
    """
    Represents a content node in the context graph.
    """
    def __init__(
        self,
        pair: 'PairGraph',
        chunk: str = None,
        node_id: int = None,
        accessors: List['AccessorNode'] = None,
        metadata: Dict[str, Any] = None,
        vector: List[float] = None,
        is_synced: bool = False,
        is_built: bool = False,
    ):
        super().__init__(
            pair=pair,
            chunk=chunk,
            node_id=node_id,
            vector=vector,
            is_synced=is_synced,
            is_built=is_built,
        )
        self._accessors = accessors
        self._metadata = metadata

    def __eq__(self, other: 'BaseNode') -> bool:
        if not isinstance(other, ContentNode):
            return False
        return self.node_id == other.node_id
    
    def __hash__(self) -> int:
        return hash('C' + str(hash(self.pair)) + self.node_id)

    # ---------------------------------------------------------------- PROPERTY METHODS ----------------------------------------------------------------

    @property
    def accessors(self):
        if not self.is_built:
            if DEBUG_OPS: print("ACCESSORS RETRIEVAL TRIGGERED BUILD FOR", self._node_id)
            self.build()
        return self._accessors

    @property
    def metadata(self):
        if not self.is_built:
            if DEBUG_OPS: print("METADATA RETRIEVAL TRIGGERED BUILD FOR", self._node_id)
            self.build()
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        if not self.is_built:
            self.build()
        self._metadata = value
        self.is_synced = False

    # ---------------------------------------------------------------- ACCESSOR MANAGEMENT ----------------------------------------------------------------

    def delete(self):
        # 1. Unlink all accessor nodes
        for accessor in self.accessors:
            accessor.unlink(self)

        # 2. Delete the node data to prevent accessing it, except the node_id so that it can still be tracked and deleted upon commit.
        self._chunk = None
        self._vector = None
        self._accessors = None

        self.is_synced = True  # Prevents pushing new data from deleted nodes
        self.is_deleted = True

        if self._node_id is None:
            raise ValueError("Cannot delete a node without a node_id set. This should never happen.")
        
    # ---------------------------------------------------------------- UTILITY -----------------------------------------------------------------

    def chunk_merge(self, chunk: str) -> str:
        """
        Returns a new merged chunk generated from the current self.chunk and the new provided chunk.
        """
        return self.pair.content_merge_function(self.chunk, chunk)

    # ---------------------------------------------------------------- COMMIT ----------------------------------------------------------------

    def _merge(self, other: 'ContentNode'):

        self.is_synced = other.is_synced

        # 0. Check if the current node is synced (if not, the merge operation will be simplified)
        if self._node_id:  # If both nodes already exist and potentially have metadata, merge them. 
            # NOTE This should be a rare case.

            # 1. Merge the chunks
            if self._chunk is not None:

                # 1. Merge the chunks
                self._chunk = self.chunk_merge(other.chunk)

                # 2. Rebuild the vector from the new chunk
                self._vector = self.pair.embedding_function(self._chunk)

                self.is_synced = False

            else:
                self._chunk = other.chunk
                self._vector = other._vector  # Using _ prevents generating the same vector twice, or generating one that would never be used because the node is an existing readonly.

            # 3. Copy node_id
            self._node_id = other.node_id

            if self._accessors is None:
                self._accessors = other.accessors

            elif self._accessors is not None:
                # 4. Merge the two accessor lists.
                diff = [acc for acc in other.accessors if acc not in self._accessors]
                self._accessors.extend(diff)

                # 5. Merge the metadatas
                if self._metadata is None:
                    self._metadata = other.metadata
                else:
                    self._metadata.update(other.metadata)

                self.is_synced = False

            else:
                self._accessors = other.accessors
                self._metadata = other.metadata

        else:  # Simplified merge operation : merge the chunks and replace all other fields of the current node with the other.
            # NOTE : This should be the most common case.

            # 1. Merge the chunks
            # Only the other chunk exists, simply use that chunk as self chunk.
            if self._chunk is None:
                # Bypass the self.chunk accessor to avoid calling the build() method
                self._chunk = other.chunk
            
            else:  # Both chunks exist : merging
                self._chunk = self.chunk_merge(other.chunk)

            # 2. Rebuild the vector from the new chunk
            self._vector = self.pair.embedding_function(self._chunk)

            # 3. Copy all of the other values from the other node into the current node
            self._node_id = other.node_id
            self._accessors = other.accessors
