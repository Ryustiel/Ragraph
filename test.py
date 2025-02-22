"""
Test script for database transactions.

This script performs the following:
  - Connects to the database (using an in-memory SQLite database for testing).
  - Creates the tables defined in the database.
  - Inserts a Content record via a ContentTransaction.
  - Inserts an Accessor record via an AccessorTransaction in the "tasks" layer.
  - Creates an edge between the inserted content and accessor with a given weight.
  - Queries the database for contents that match a query string ("Demo") and displays the content text along with the edge weight.
"""

from sqlalchemy.orm import Session
from src.database import Content, DatabaseConnection
from src.transactions import ContentTransaction, AccessorTransaction

import os
import dotenv
dotenv.load_dotenv(override=True)


db_conn = DatabaseConnection(url=os.environ.get("DATABASE_CONNECTION_STRING"))
db_conn.drop_tables()
db_conn.create_tables()

session: Session = db_conn.create_session()

try:
    content_tx = ContentTransaction(text="Demo Content Inserted")
    content_tx.commit(session)
    print("Inserted Content ID:", content_tx.id)
    
    accessor_tx = AccessorTransaction(embeddable="Demo Accessor", layer_name="tasks")
    accessor_tx.commit(session)
    print("Inserted Accessor ID:", accessor_tx.id)
    
    accessor_tx.set_edge(session, content_tx, weight=0.85)
    print("Edge added between Accessor and Content")
    
    session.commit()
    

    results = session.query(Content).filter(Content.text.like("%Demo%")).all()
    print("\nQuery Results:")

    for content_obj in results:

        linked_edges = list(content_obj.get_linked_accessors(allowed_layers=["tasks"]))
        for edge in linked_edges:
            print("Content:", content_obj.text, "Edge Weight:", edge.weight)
finally:
    session.close()
