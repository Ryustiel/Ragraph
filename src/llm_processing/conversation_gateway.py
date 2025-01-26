"""
Produces contents from an input ChatHistory.
Also handles updating a buffer and generating updates to the contents and weights.
"""

from typing import (
    List,
)

from ..database import ContentSet
from raphlib import ChatHistory


class ConversationGateway:

    @classmethod
    def from_chat_history(cls, history: ChatHistory) -> 'ConversationGateway':
        pass

    def get_contents(self) -> List[str]:
        """
        Gets the relevant pieces of content from the conversation history.
        Also updates the review buffer to keep track of this new content retrieval operation.
        """
        pass

    def review_cascade(self):
        
        # 1. The passive review should occur on random chunks in the buffer once a couple messages are before and after the current message
        # 2. An active review should occur every time a couple of messages have been received in the buffer after a target message. 
        # Let's say 10 or a couple time without a new message.
        # 3. Both reviews should trigger a weight update.
        # 4. Upon running a content add operation (following this kind of upgrade), 
        # the database should be checked for latent chunks that are numerous for a single content node.
        # if many are found, the latent chunks can be merged.
        pass
