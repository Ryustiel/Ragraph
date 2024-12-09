"""
PairManager is the base object that interacts with the sql database.
"""

import warnings
import chromadb
from typing import Callable, List, Dict, Any, Tuple, Callable, Generator, Optional

# Suppress FutureWarnings from transformers (temporary fix)
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')

from ..pairs import PairGraph, NodePlot, EdgePlot
from ..nodes import AccessorNode, ContentNode

DEBUG = False


# =================================================================  PAIR GRAPH  ==============================================================


class ChromaGraph(PairGraph):
    """
    Represents a pair of chroma collections referred to as "Accessors" and "Contents".
    Each pair is uniquely identified by a `name` and can be accessed and edited through this object.

    Unlike PairGraph, this class does not expose a lot of methods for interacting with the nodes.
    It only provides rudimentary operations and manages the collections themselves.
    """
    def __init__(self, 
                 client: chromadb.ClientAPI,  
                 pair_name: str, 
                 embedding_function: Callable[[str], List[float]],
                 content_merge_function: Optional[Callable[[str], str]] = None,
                 accessor_merge_function: Optional[Callable[[str], str]] = None,
                 content_similarity_threshold: float = 0.5,
                 accessor_similarity_threshold: float = 0.5,
                 content_equality_threshold: float = 1,
                 accessor_equality_threshold: float = 1,
        ):
        """
        Initializes the PairManager with a unique pair name and sets up the corresponding collections.

        Args:
            client (chromadb.ClientAPI): The ChromaDB client.
            pair_name (str): Unique name for the pair of collections.
            embedding_function (Callable[[str], List[float]]): Embedding function for the collection.
        """
        super().__init__(
            pair_name=pair_name, 
            embedding_function=embedding_function, 
            content_merge_function=content_merge_function, 
            accessor_merge_function=accessor_merge_function,
            content_similarity_threshold=content_similarity_threshold,
            accessor_similarity_threshold=accessor_similarity_threshold,
            content_equality_threshold=content_equality_threshold,
            accessor_equality_threshold=accessor_equality_threshold,
        )

        self.client = client

        # Define collection names based on name
        self.accessors_collection_name = f"Accessors_{pair_name}"
        self.contents_collection_name = f"Contents_{pair_name}"

        # Initialize collections
        self.accessors = self._get_or_create_collection(self.accessors_collection_name)
        self.contents = self._get_or_create_collection(self.contents_collection_name)

        # Maintain separate ID counters for each collection
        self.max_accessor_id = self._compute_current_collection_max_id(self.accessors)
        self.max_content_id = self._compute_current_collection_max_id(self.contents)

    def _get_or_create_collection(self, collection_name: str):
        """
        Retrieves an existing collection or creates a new one if it does not exist.

        Args:
            collection_name (str): Name of the collection.

        Returns:
            chromadb.Collection: The retrieved or newly created collection.
        """
        existing_collections = self.client.list_collections()
        for collection in existing_collections:
            if collection.name == collection_name:
                return self.client.get_collection(collection_name)
        # If not found, create a new collection with appropriate embedding function
        if self.embedding_function is None:
            raise Exception(f"""Could not find the collection \"{collection_name}\". 
                            Proceeded to create it but missing an embedding function. 
                            Provide it in the PairManager constructor for the pair with that collection name.""")
        return self.client.create_collection(name=collection_name)
    
    # ---------------------------------------------------------------- ID MANAGEMENT ----------------------------------------------------------------

    def _compute_current_collection_max_id(self, collection: chromadb.Collection) -> int:
        """
        Retrieves the maximum integer ID from a collection.
        This function is expensive, use get_

        Args:
            collection (chromadb.Collection): The collection to query.

        Returns:
            int: The maximum ID found, or -1 if the collection is empty.
        """
        results = collection.get(
            limit=1000000  # Adjust as needed for larger datasets
        )
        if not results['ids']:
            return -1
        # Extract integer IDs
        ids = [int(id_) for id_ in results['ids']]
        return max(ids) if ids else -1
    
    def get_next_accessor_id(self) -> int:
        """
        Returns the next accessor ID for a node to be added to the accessor collection.
        """
        self.max_accessor_id += 1
        return self.max_accessor_id
    
    def get_next_content_id(self) -> int:
        """
        Returns the next content ID for a node to be added to the content collection.
        """
        self.max_content_id += 1
        return self.max_content_id
    
    # ---------------------------------------------------------------- NODES ----------------------------------------------------------------------

    def Accessor(self, input: str, is_node_id: bool = False):
        """
        Create a new accessor node object from the provided data.
        """
        if is_node_id:
            node = ChromaAccessor(self, node_id=input)
        else:
            node = ChromaAccessor(self, chunk=input)

        self.register(node)
        return node

    def Content(self, input: str, is_node_id: bool = False):
        """
        Create a new content node object from the provided data.
        """
        if is_node_id:
            node = ChromaContent(self, node_id=input)
        else:
            node = ChromaContent(self, chunk=input)
        self.register(node)
        return node

    # ---------------------------------------------------------------- PLOT --------------------------------------------------------------------

    def get_nodes(self) -> Generator[NodePlot, None, None]:
        """
        Produce the list of nodes in the current graph.
        """
        accessors = self.accessors.get(include=["documents"])
        contents = self.contents.get(include=["metadatas", "documents"])

        # TODO : When AI Labeling, run the functions as batches.
        for id, document in zip(accessors["ids"], accessors["documents"]):
            yield NodePlot(
                id = f"accessor_{id}",
                type = "accessor",
                chunk = document,
                label = "" # LLMFunction(LLM, f"Find 2 words as \"word1\\nword2\" to summarize this text: {document}", summary=str).invoke().summary
            )

        for id, metadata, document in zip(contents["ids"], contents["metadatas"], contents["documents"]):
            yield NodePlot(
                id = f"content_{id}",
                type = "content",
                chunk = document,
                label = "" # LLMFunction(LLM, f"Find 2 words as \"word1\\nword2\" to summarize this text: {document}", summary=str).invoke().summary
            )

    def get_edges(self) -> Generator[EdgePlot, None, None]:
        """
        Produce the list of edges in the current graph.
        """

        accessors_ids = self.accessors.get(include=list())["ids"]

        accessors = [self.Accessor(node_id, is_node_id=True) for node_id in accessors_ids]

        for acc in accessors:

            for content in acc.contents:
                yield EdgePlot(
                    start = f"accessor_{acc.node_id}",
                    end =  f"content_{content.node_id}",
                    type = "child",
                    label = ""
                )

            for neighbor, weight in acc.neighbors.items():
                EdgePlot(
                    start = f"accessor_{acc.node_id}",
                    end = f"accessor_{neighbor.node_id}",
                    type = "neighbor",
                    label = str(weight / len(acc.contents))
                )

    # ---------------------------------------------------------------- PERSISTANCE ----------------------------------------------------------------
    
    def commit(self):
        super().commit()
        # self.persist()

    def persist(self):
        """
        Persists the current state of the database to disk.
        """
        self.client.persist()

    def close(self):
        """
        Closes the client and ensures all data is persisted.
        """
        self.persist()
        self.client.close()
    



# ================================================================= ACCESSOR =================================================================




class ChromaAccessor(AccessorNode):
    """
    An accessor node in the ragraph. The operations rely on ChromaDB and the chroma accessor.
    """
    def __init__(
        self,
        pair: 'ChromaGraph',
        chunk: str = None,
        node_id: int = None,
        contents: List['ContentNode'] = None,
        neighbors: Dict['AccessorNode', int] = None,
        vector: List[float] = None,
        is_synced: bool = False,
        is_built: bool = False,
    ):
        super().__init__(pair, chunk, node_id, contents, neighbors, vector, is_synced, is_built)
        self.pair = pair

    def _extract_neighbors_and_contents(self, metadata: Dict[str, Any]) -> Tuple[List['ChromaContent'], Dict['ChromaAccessor', int]]:
        """
        Import the contents and neighbors from the metadata search results.
        """
        next_key = "next"

        # 1. Import the contents
        if not next_key in metadata.keys():
            contents = list()
            neighbors = dict()
        else:
            if not isinstance(metadata[next_key], str):
                raise ValueError(f"The {next_key} field in the metadata must be a string, got {type(metadata[next_key])}")
            contents = [ChromaContent(node_id=id_, pair=self.pair) for id_ in metadata[next_key].split(',') if id_]

            neighbors = {AccessorNode(node_id=node_id): weight for node_id, weight in metadata.items() if node_id != next_key}

        return contents, neighbors

    def query(self, max_results: int = 1, similarity_threshold: float = None) -> Generator['ChromaAccessor', None, None]:

        if similarity_threshold is None:
            similarity_threshold = self.pair.accessor_similarity_threshold

        results = self.pair.accessors.query(query_embeddings=self.vector, n_results=max_results, include=["metadatas", "documents", "embeddings", "distances"])
        for node_id_list, chunk_list, embedding_list, metadata_list, distance_list in zip(results["ids"], results["documents"], results["embeddings"], results["metadatas"], results["distances"]):
            
            # All of the values are in the shape of list of length 1, because there is only 1 vector input.
            if distance_list and distance_list[0] < similarity_threshold:  # This also handles the case where there is no embedding in the database (distance == list())

                contents, neighbors = self._extract_neighbors_and_contents(metadata_list[0])

                yield ChromaAccessor(
                    pair=self.pair,
                    node_id=node_id_list[0],
                    chunk=chunk_list[0],
                    contents=contents,
                    neighbors=neighbors,
                    vector=embedding_list[0],
                    is_synced=True,
                    is_built=True,
                )

    def query_equal(self) -> Optional['ChromaAccessor']:
        """
        Find the node that is the most similar to the current node
        """
        results = self.query(max_results=1, similarity_threshold=self.pair.accessor_equality_threshold)
        return next(results, None)

    def _as_new(self):
        """
        Initialize a brand new state of the object (locally).
        """
        if self._chunk is None:
            raise ValueError("Cannot create an AccessorNode without a chunk.")

        self._node_id = str(self.pair.get_next_accessor_id())
        self._chunk = self._chunk
        self._contents = list()
        self._neighbors = dict()
        self.is_synced = False

    def _from_id(self) -> bool:
        """
        Update itself by fetching a node in the database using the node_id.

        Returns:
            bool: Whether the update was successful (or could be initiated at all).
        """
        if self._node_id is None:
            return False

        results = self.pair.accessors.get(ids=[self.node_id], include=["metadatas", "documents", "embeddings"])
        if results:
            metadata = results["metadatas"][0]
            contents, neighbors = self._extract_neighbors_and_contents(metadata)

            self._node_id = self.node_id
            self._chunk = results["documents"][0]
            self._contents = contents
            self._neighbors = neighbors
            self._vector = results["embeddings"][0]
            self.is_synced = True
            self.is_built = True

            return True

        else:
            return False

    def database_update(self):
        """
        Update itself in the database.
        """
        if DEBUG: print(f"Updating node {self.node_id} on the database.")

        next_key = "next"

        metadata = self.neighbors.copy()
        metadata[next_key] = ",".join(content.node_id for content in self.contents)
        
        self.pair.accessors.upsert(
            ids=[self.node_id], 
            embeddings=[self.vector],
            metadatas=[metadata],
            documents=[self.chunk]
        )

    def database_delete(self):
        """
        Delete the node for itself from the database.
        """
        if DEBUG: print(f"Deleting node {self.node_id} on the database.")

        if self._node_id is None or self.is_deleted:
            raise ValueError(f"Node {self.node_id} was deleted locally before being deleted from the database.")
        
        self.pair.contents.delete(ids=[self.node_id])




# ================================================================= CONTENT =================================================================




class ChromaContent(ContentNode):
    """
    A content node in the graph. Performs operations based on a ChromaDB connection.
    """
    def __init__(
        self,
        pair: 'ChromaGraph',
        chunk: str = None,
        node_id: int = None,
        accessors: List['AccessorNode'] = None,
        metadata: Dict[str, Any] = None,
        vector: List[float] = None,
        is_synced: bool = False,
        is_built: bool = False,
    ):
        super().__init__(pair, chunk, node_id, accessors, metadata, vector, is_synced, is_built)
        self.pair = pair

    def _extract_accessors_and_metadata(self, metadata: Dict[str, Any]) -> Tuple[List['ContentNode'], Dict['AccessorNode', int]]:
        """
        Import the contents and neighbors from the metadata search results.
        """
        next_key = "next"

        # 1. Import the contents
        if not next_key in metadata.keys():
            accessors = list()
            data = dict()
        else:
            if not isinstance(metadata[next_key], str):
                raise ValueError(f"The {next_key} field in the metadata must be a string, got {type(metadata[next_key])}")
            accessors = [ChromaAccessor(node_id=id_, pair=self.pair) for id_ in metadata[next_key].split(',') if id_]
            
            data = {key: value for key, value in metadata.items() if key != next_key}

        return accessors, data

    def query(self, max_results: int = 1, similarity_threshold: float = None) -> Generator['AccessorNode', None, None]:

        if similarity_threshold is None:
            similarity_threshold = self.pair.content_similarity_threshold

        results = self.pair.contents.query(query_embeddings=[self.vector], n_results=max_results, include=["metadatas", "documents", "embeddings", "distances"])
        for node_id_list, chunk_list, embedding_list, metadata_list, distance_list in zip(results["ids"], results["documents"], results["embeddings"], results["metadatas"], results["distances"]):

            # All of the values are in the shape of list of length 1, because there is only 1 vector input.
            if distance_list and distance_list[0] < similarity_threshold:  # This also handles the case where there is no embedding in the database (distance == list())

                accessors, metadata = self._extract_accessors_and_metadata(metadata_list[0])
                
                yield ChromaContent(
                    pair=self.pair,
                    node_id=node_id_list[0],
                    chunk=chunk_list[0],
                    accessors=accessors,
                    metadata=metadata,
                    vector=embedding_list[0],
                    is_synced=True,
                    is_built=True,
                )

    def query_equal(self) -> Optional['ChromaContent']:
        """
        Find the node that is the most similar to the current node
        """
        results = self.query(max_results=1, similarity_threshold=self.pair.content_equality_threshold)
        return next(results, None)

    def _as_new(self):
        """
        Initialize a brand new state of the object (locally).
        """
        if self._chunk is None:
            raise ValueError("Cannot create an AccessorNode without a chunk.")

        self._node_id = str(self.pair.get_next_content_id())
        self._chunk = self._chunk
        self._accessors = list()
        self._metadata = dict()
        self.is_synced = False

    def _from_id(self) -> bool:
        """
        Update itself by fetching a node in the database using the node_id.

        Returns:
            bool: Whether the update was successful (or could be initiated at all).
        """
        if self._node_id is None:  # If no node_id is specified can't find anything in the database
            return False

        results = self.pair.contents.get(ids=[self.node_id])
        if results:
            metadata = results["metadatas"][0]
            accessors, data = self._extract_accessors_and_metadata(metadata)

            self._node_id=self.node_id
            self._chunk=results["documents"][0]
            self._accessors=accessors
            self._metadata=data
            self._vector=results["embeddings"][0]
            self.is_synced=True

            return True

        else: # Did not find anything in the database
            return False

    def database_update(self):
        """
        Update itself in the database.
        """
        if DEBUG: print(f"Updating node {self.node_id} in the database.")

        next_key = "next"

        metadata = self.metadata.copy()
        metadata[next_key] = ",".join(accessor.node_id for accessor in self.accessors)

        self.pair.contents.upsert(
            ids=[self.node_id], 
            embeddings=[self.vector],
            metadatas=[metadata],
            documents=[self.chunk]
        )

    def database_delete(self):
        """
        Delete the node for itself from the database.
        """
        if DEBUG: print(f"Deleting node {self.node_id} in the database.")

        if self._node_id is None or self.is_deleted:
            raise ValueError(f"Node {self.node_id} was deleted locally before being deleted from the database.")
        
        self.pair.contents.delete(ids=[self.node_id])

