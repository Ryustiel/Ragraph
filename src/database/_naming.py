
ACCESSOR_ATTRIBUTE_PREFIX = "accessors_"

CONTENT_TABLE_NAME = "context__contents"
CLUSTER_TABLE_NAME = "context__clusters"
REVIEW_TABLE_NAME = "context__reviews"
CONTENT_LINK_TABLE_NAME = "context__content_links"
LATENT_CHUNKS_TABLE_NAME = "context__latent_chunks" 
HYPER_CLUSTER_TABLE_NAME = "context__hyper_clusters"
LATENT_REVIEW_TABLE_NAME = "context__latent_reviews"

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
