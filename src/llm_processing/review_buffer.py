
from typing import (
    List,
)

from raphlib import ChatHistory, ChatMessage


class TrackedMessage(ChatMessage):
    """
    A message with tracking of RAG operations.
    """
    content_ids: List[int]  # The ids of the content nodes that were used in this message


class ReviewBuffer(ChatHistory):
    """
    An extension of chat history that will only store a few messages
    and expose a couple of utility methods to handle reviewing chat messages.
    """
    processed_message_index = 0  # Index of the message to process in the history

