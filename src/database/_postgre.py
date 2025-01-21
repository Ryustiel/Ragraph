"""
Handle engine initialization, table creation etc.
"""
import os

from typing import (
    Any,
    Generator,
    Optional,
)

from sqlalchemy import create_engine, text, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import ProgrammingError

# NOTE : Keep these imports ordered in the order the Base object is juggled around. (case of adding an extra table)
from .contents import (
    Content,
    Accessor,
    ACCESSORS,
)
from .prompts import (
    UserStatement,
    SuggestedReply,
    Base,
)


class DatabaseConnection:
    """
    Authenticates and exposes methods to interact with the Seiso database.
    This object can be used in a context manager.
    Calling it will return a generator whose 1st iteration will yield a session, and 2nd iteration will close the session.
    """
    def __init__(self, url: Optional[str] = None, autocommit = False):

        if url is None:
            url = os.environ.get("DATABASE_CONNECTION_STRING")
            if url is None:
                raise ValueError("No database connection string provided. DATABASE_CONNECTION_STRING might be missing from the environment.")
            
        self.url = url
        self._engine = None
        self._sessionmaker = None
        self._active_context_session = None

        self.autocommit = autocommit
        
    @property
    def engine(self) -> Engine:
        """
        Creates and returns the engine if it's not already.
        """
        if self._engine is None:
            self._engine = create_engine(self.url)
        return self._engine
    
    # ================================================================= MAINTAINANCE OPERATIONS
    
    def create_database(self):
        """Creates the database corresponding to the base_url if it does not exist already."""

        url = make_url(self.url)
        database_name = url.database
        if not database_name:
            raise ValueError("No database name found in the connection string.")

        # Replace database name with the "maintenance" database (e.g., 'postgres') to connect to the server
        url = url.set(database=None)
        server_url = str(url)

        # Use a connection to the server to create the database
        temp_engine = create_engine(server_url)
        try:
            with temp_engine.connect() as connection:
                # Check if the database already exists
                result = connection.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{database_name}';"))
                database_exists = result.scalar() is not None

                if not database_exists:
                    connection.execute(text(f'CREATE DATABASE "{database_name}";'))
                    print(f"Database '{database_name}' created successfully.")
                else:
                    print(f"Database '{database_name}' already exists.")
        except ProgrammingError as e:
            raise RuntimeError("Failed to create the database. Ensure you have the necessary privileges.") from e
        finally:
            temp_engine.dispose()

        self._engine = None  # Now reinitialize the engine to point to the newly created database

    def create_tables(self):
        """Creates all tables in the database."""
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        """Drops all tables in the database."""
        Base.metadata.drop_all(self.engine)

    def activate_pgvector(self):
        with self as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            if not self.autocommit: session.commit()

    # ================================================================= SESSION HANDLING

    def create_session(self) -> Session:
        if self._sessionmaker is None:
            self._sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        return self._sessionmaker()

    def get_session(self) -> Generator[Session, Any, None]:
        """
        Provide a generator type of session handler.
        The 1st iteration will yield a session, and the 2nd iteration will close the session.
        """
        session = self.create_session()
        try:
            yield session
        finally:
            session.close()

    def __call__(self, autocommit = False) -> Generator[Session, Any, None]:
        """
        Provide a generator type of session handler.
        The 1st iteration will yield a session, and the 2nd iteration will close the session.
        """
        return self.get_session(autocommit=autocommit)
    
    def __enter__(self) -> Session:
        self._active_context_session = self.create_session()
        return self._active_context_session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.autocommit:
            self._active_context_session.commit()
        self._active_context_session.close()
        return False  # Handle exceptions if necessary; return True to suppress them, False to propagate
