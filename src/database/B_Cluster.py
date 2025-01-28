
from typing import (
    List,
)

from sqlalchemy import (
    Column, 
    Integer, 
    String, 
    ForeignKey,
    Table,
)
from sqlalchemy.orm import relationship

from ._naming import LATENT_CHUNKS_TABLE_NAME, CONTENT_TABLE_NAME, CLUSTER_TABLE_NAME, HYPER_CLUSTER_TABLE_NAME
from .A_Edge import *


class LatentChunk(Base):
    """
    When many text inputs fall in the similarity_condition and not the identity_condition
    of a content node, they are stored in this table in reference to the node they were most similar with.

    When they accumulate for a particular content node, they are merged together using a LLM function
    and replace the text and vector of the content node, effectively replacing it with "more relevant" content.
    """
    __tablename__ = LATENT_CHUNKS_TABLE_NAME
    
    # Most similar content id
    id = Column(Integer, primary_key=True, autoincrement=True)
    content_id = Column('content_id', Integer, ForeignKey(f'{CONTENT_TABLE_NAME}.id', ondelete="CASCADE"), nullable=False)
    text = Column('text', String, nullable=False)

    content = relationship('Content', back_populates="latent_chunks")


hypercluster_cluster_association = Table(
    f'{HYPER_CLUSTER_TABLE_NAME}_association',
    Base.metadata,
    Column('hypercluster_id', String, ForeignKey(f'{HYPER_CLUSTER_TABLE_NAME}.id', ondelete="CASCADE")),
    Column('cluster_id', Integer, ForeignKey(f'{CLUSTER_TABLE_NAME}.id', ondelete="CASCADE"))
)


class Cluster(Base):
    """
    Stores a chunk that represents the aggregated data from multiple contents.
    This allows for shorter contents in prompts and less redundancy.
    """
    __tablename__ = CLUSTER_TABLE_NAME

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column('text', String, nullable=True)  # The cluster's text content.

    hyperclusters = relationship(
        "HyperCluster",
        secondary=hypercluster_cluster_association,
        back_populates="clusters"
    )
    contents = relationship('Content', back_populates="cluster")

    def has_content(self) -> bool:
        """
        Returns True if the cluster has any content that can replace the content nodes, False otherwise.
        """
        return self.text is not None


class HyperCluster(Base):
    """
    Stores aggregated data from multiple clusters.
    You can search for any set of clusters and get the aggregated data.
    """
    __tablename__ = HYPER_CLUSTER_TABLE_NAME

    id = Column(String, primary_key=True)
    text = Column('text', String, nullable=True)  # The hypercluster's text content.

    clusters = relationship(
        "Cluster",
        secondary=hypercluster_cluster_association,
        back_populates="hyperclusters"
    )

    @classmethod
    def from_ids(cls, session: Session, ids: List[int]) -> "HyperCluster":
        """
        Fetch a cluster corresponding to the input set of cluster ids.
        """
        # Sort IDs to ensure consistent hyper_id generation.
        sorted_ids = sorted(ids)
        hyper_id = "".join(map(str, sorted_ids))
        
        # Check if the HyperCluster already exists.
        hypercluster = session.query(cls).filter_by(id=hyper_id).first()
        if hypercluster:
            print(f"Fetched existing hypercluster with id: {hyper_id}")
            return hypercluster
        
        # Create a new HyperCluster
        clusters = session.query(Cluster).filter(Cluster.id.in_(ids)).all()  # Fetch clusters by IDs
        if len(clusters) != len(ids):
            raise ValueError("Some ids did not correspond to an existing cluster")

        hypercluster = cls(id=hyper_id, clusters=clusters)
        session.add(hypercluster)
        session.commit()
        print(f"Created new hypercluster with id: {hyper_id}")
        return hypercluster
