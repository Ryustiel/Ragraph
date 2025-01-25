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

from typing import (
    List, 
    Dict,
    Any,
    ClassVar,
    Generator,
)

from sqlalchemy import (
    Column, Integer, String, 
    Float, ForeignKey, Table, 
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship, declarative_base, Session
from sqlalchemy.ext.declarative import declared_attr

from ._vectorizer import Vectorizer
from ._config import (
    CONTENT_SIMILARITY_THRESHOLD, 
    CONTENT_IDENTITY_THRESHOLD,
    ACCESSOR_CONFIG, 
    AccessorConfig,
)

Base = declarative_base()
ACCESSORS: Dict[str, 'Accessor'] = {}

# ================================================================= CONSTANTS

ACCESSOR_ATTRIBUTE_PREFIX = "accessors_"

CONTENT_TABLE_NAME = "context__contents"
LATENT_CHUNKS_TABLE_NAME = "context__latent_chunks"

def accessor_table_name(label: str) -> str:
    """Generates a name for an accessor table"""
    return "context_" + label

def accessor_class_name(label: str) -> str:
    """Generates a name for a class that will be used to define a database model for an accessor"""
    return f"Context{label.capitalize()}Accessor"

def edge_table_name(label: str) -> str:
    """Generates a name for an accessor edge table"""
    return accessor_table_name(label) + "_edges"

def edge_class_name(label: str) -> str:
    """Generates a name for a class that will be used to define a database model for the edges of an accessor"""
    return accessor_class_name(label) + "Edge"

def attribute_name(label: str) -> str:
    """Generate an attribute name for referencing, in a Content table, all the accessors nodes of a particular layer that are related to that table"""
    return ACCESSOR_ATTRIBUTE_PREFIX + label

# ================================================================= DATA TRANSPORT

class ContentSet(Dict['Content', float]):
    """
    A set of content nodes. This class is used to implement repetitive operations on sets of nodes.
    Keys are the ids of the contents in the database, and values are their associated weights.
    """
    def __sum__(self, other: 'ContentSet') -> 'ContentSet':
        """
        Summing two content sets results in a new set with the union of contents, 
        with new weights representing the combination of both.
        """
        for key, other_weight in other.items():
            self[key] = max(self[key], other_weight)

class ContextInput(Dict[str, str]):
    """
    Lets you define the current context and extract content nodes from the various accessor layers based on it.
    The structure is {<accessor layer name>: <context input>}
    """
    pass

# ================================================================= EDGES

class AccessorEdge(Base):
    """
    This is a base class for edges between accessors and contents.
    This one is abstract and needs to be subclassed as is below.
    """
    __abstract__ = True

    weight = Column('weight', Float, nullable=False, default=0.5)
    content_id = Column('content_id', Integer, ForeignKey(f'{CONTENT_TABLE_NAME}.id'), primary_key=True)

    @declared_attr
    def accessor_id(cls):
        """The id of an accessor"""
        return 0

    @declared_attr
    def accessor(cls):
        """The accessor this edge connects to"""
        return Accessor()
    
    @declared_attr
    def content(cls):
        """The content this edge connects to"""
        return Content()

# ================================================================= CHUNK STORAGE

class LatentChunk(Base):
    """
    When many text inputs fall between the similarity_threshold and the identity_threshold
    of a content node, they are stored in this table in reference to the node they were most similar with.

    When they accumulate for a particular content node, they are merged together using a LLM function
    and replace the text and vector of the content node, effectively replacing it with "more relevant" content.
    """
    __tablename__ = LATENT_CHUNKS_TABLE_NAME
    
    # Most similar content id
    id = Column(Integer, primary_key=True, autoincrement=True)
    content_id = Column('content_id', Integer, ForeignKey(f'{CONTENT_TABLE_NAME}.id', ondelete="CASCADE"), nullable=False)
    text = Column('text', String, nullable=False)

# ================================================================= CONTENT AND ACCESSOR

class Content(Base):
    """Contains the methods that will be implemented in the contents class."""

    __tablename__ = CONTENT_TABLE_NAME
    
    # TODO : Use Mapped for typing in here.

    id = Column('id', Integer, primary_key=True, autoincrement=True)
    text = Column('text', String, nullable=False)
    embedding = Column('embedding', Vector(dim=3072), nullable=False)
    behavior = Column('behavior', String(20), nullable=False, default='regular')  # Special flag for the content, has no use for now

    latent_chunks = relationship('LatentChunk', backref="content")

    def __hash__(self):
        """Unique hash for indexing in a ContentSet"""
        return hash(self.id)

    @classmethod
    def from_text(cls, text: str, context: ContextInput = None) -> 'Content':
        """
        Create a vector from a text string

        1. Compute the context.
        2. Checks if the content already exists (by running a vector search)
        3. If below the similarity threshold, then create a new content linked to each of the accessor of the context
        4. If above the similarity threshold and below the identity threshold 
        then save the chunk as latent content and link the accessors from the context who do not have a link to the current context, 
        with the default initialization weight.
        5. If above the identity threshold, simply link the accessors from the context who do not have a link to the identical content node.
        6. Return that new or updated node.

        Updates to the weight are caused by the other mechanisms (retrieval evaluation | inquiry events)
        """
        embedding = Vectorizer.process(text)
        return cls(text=text, embedding=embedding)
    
    # NOTE : After that we need a "conversation buffer" + an evaluation object

    @classmethod
    def find_content(cls, session: Session, content: str, max_output: int = 10) -> List['Content']:
        """
        Runs a vector search for content similar to the current one.
        """
        input_vector = Vectorizer.process(content)
        return (
            session.query(cls)
            .filter(cls.embedding != None)  # Exclude entries without vectors
            .order_by(cls.embedding.l2_distance(input_vector))  # L2 distance (Euclidean distance)
            .limit(max_output)
            .all()
        )
    
    @property
    def linked_accessors(self) -> Generator[List[AccessorEdge], Any, None]:
        """The list of accessors that were linked to the current content."""
        # look at all the attributes whose name begin with ACCESSOR_ATTRIBUTE_PREFIX
        # yield the values of the attribute
        for attribute_name, value in self.__dict__.items():
            if attribute_name.startswith(ACCESSOR_ATTRIBUTE_PREFIX):
                yield value  # If an accessor attribute, then it contains the list of links to these accessors

    # NOTE : Contents should always be added via an accessor, not the other way around. No "accessor append" in this class.
    
    def through_weight_formula(self, link_weight_to_acessor: float, link_weight_to_content: float) -> float:
        """Calculate the weight of a content accessed through / via an intermediate accessor"""
        return link_weight_to_acessor * link_weight_to_content
    
    # TODO : Find related accessors + weights
    def next_contents(self) -> ContentSet:
        """
        Find the content nodes that can be reached indirectly from the current content node.

        1. Look for the accessors linked to the current content and retain the weight of the links
        2. Get the contents attached to each of these accessors and compute a new weight link 
        based on the weights of both the [current -> intermediate accessor] and [intermediate -> other content]
        3. Compile the weights of each accessor in a ContentSet
        4. Merge the content sets (which also retains the max weight of each accessor)
        5. Return the merged content set, effectively representing the content nodes that can be reached indirectly from the current content node.
        """
        sets = []
        for links in self.linked_accessors:
            for link in links:
                local_set = ContentSet()
                for content_link in link.accessor.contents:
                    local_set[content_link.content.id] = self.through_weight_formula(link.weight, content_link.weight)
                sets.append(local_set)
        return sum(sets) if sets else ContentSet()
    

class Accessor(Base):
    """
    This base class for accessors.
    This one is abstract and needs to be subclassed as is below.
    """
    __abstract__ = True

    edges = ClassVar['AccessorEdge']
    layer_name = ClassVar[str]
    accessor_config = ClassVar[AccessorConfig]

    id = Column(Integer, primary_key=True, autoincrement=True)

    @declared_attr
    def embedding(cls):
        """
        This is a placeholder for the embedding attribute,
        which will be of varying length depending on the accessor class.
        """
        return Column(Vector(dim=0)) # Vector slot for the content
     
    def get_contents(self) -> ContentSet:
        """
        Returns a set of weighted contents.
        """
        pass

    @classmethod
    def from_text(cls, session: Session, text: str) -> 'Accessor':
        """
        1. Finds the most similar accessor in the vector db.
        2. If the closest match is below the similarity threshold, return the accessor.
        3. Otherwise, create a new accessor in the layer with the input text as a content.
        """
        # Find the most similar accessor in the vector db
        similar_accessors = cls.vector_search(session, Vectorizer.process(text), max_output=1)
        
        if similar_accessors:
            # If the closest match is below the similarity threshold, return the accessor
            if similar_accessors[0].embedding.l2_distance(Vectorizer.process(text)) > config.similarity_threshold:
                return similar_accessors[0]
            else:
                # Otherwise, create a new accessor in the layer with the input text as a content
                return cls(embedding=Vectorizer.process(text))
        else:
            # If no similar accessor is found, create a new accessor in the layer with the input text as a content
            return cls(embedding=Vectorizer.process(text))
    
    @classmethod
    def vector_search(cls, session: Session, input_vector: List[float], max_output: int = 1) -> List['Accessor']:
        """
        Runs a vector search for content similar to the current one.
        """
        return (
            session.query(cls)
            .filter(cls.embedding != None)  # Exclude entries without vectors
            .order_by(cls.embedding.l2_distance(input_vector))  # L2 distance (Euclidean distance)
            .limit(max_output)
            .all()
        )

# ================================================================= DEFINING ACCESSOR SPECIALIZED ITEMS

for layer_name, config in ACCESSOR_CONFIG.items():
    
    accessor_edges = type(
        edge_class_name(layer_name),
        (AccessorEdge,),
        {
            "__tablename__": edge_table_name(layer_name),
            "accessor_id": Column(
                'accessor_id', 
                Integer, 
                ForeignKey(f'{accessor_table_name(layer_name)}.id'), 
                primary_key=True,
            ),
            "content": relationship(
                "Content", 
                backref=attribute_name(layer_name),
            ),
            "accessor": relationship(
                accessor_class_name(layer_name),
                backref="contents",
            ),
        },
    )
    
    accessor_class = type(
        accessor_class_name(layer_name),
        (Accessor,),
        {
            "__tablename__": accessor_table_name(layer_name),
            "embedding": Column(
                Vector(dim=config.embeddings_dimension),
                nullable=False,
            ),
            "edges": accessor_edges,
            "layer_name": layer_name,
            "accessor_config": config, 
        },
    )

    ACCESSORS[layer_name] = accessor_class
