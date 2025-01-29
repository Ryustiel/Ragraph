
from typing import (
    List,
    Optional,
    Callable,
)
from pydantic import BaseModel

from ..database import ContentSet, DatabaseConnection, Context, ACCESSORS
from .review_buffer import ReviewBuffer
from ._llm import LLM

from raphlib import ChatHistory, ChatMessage, LLMFunction

class ContextInput(BaseModel):
    action: Optional[str] = None
    tone: Optional[str] = None

CREATE_CONTEXT_PROMPT = LLMFunction(LLM,
                    """
                    Based on the provided conversation, 
                    populate the fields of your structured output with keywords and phrases that best represent the state of the conversation
                    especially around the last user message. You may leave any field blank (empty string) if there's nothing to put in here.
                    \nDetails on some of the fields:
                    * "action" represents what somebody should do to fulfill the user request. This should only contain verbal context, not the exact thing to do ("Search the web", "Make a joke", "Shout", ...) and NOT ("Search the web for", "Make a joke about", ...)
                    * "tone" represents how the user seems to feel like. Can be a very detailed description. ("A bit bored", "Excited", "Conspicious",...)
                    \n\nCONVERSATION: {conversation}
                    """,
                    pydantic_model = ContextInput,
                )


class ConversationGateway:
    """
    0. This object will receive a ChatHistory via the process method, update and check the ReviewBuffers and the LatentChunks.
    1. It will then handle the history to a LLMFunction to extract a ContextInput.
    2. Then create a ReviewBuffer object with (Backward Context - Sentence Anchor - Context Input)
    3. Then get the contents via the Context and return a list of contents.
    """
    def __init__(self):
        self.history = ChatHistory()
        self.pending_reviews: List[ReviewBuffer] = []  # Stores the reviews that do not have enough context to complete

    def add_message_no_output(self, message: ChatMessage):
        """
        Adds the message to the conversation gateway without computing the contents.
        This is useful when populating the history with initial messages, 
        and when inserting AI responses that do not need to be associated with a content set.
        """

        self.history.append(message)
        for i, review in reversed(list(enumerate(self.pending_reviews))):
            review.add_message(message)
            if review.is_complete():
                self.pending_reviews.pop(i)

        self.history = self.history.last(5)  # Only retain the last 5 messages

    def add_message(self, message: ChatMessage) -> ContentSet:
        """
        Stores the message and updates the review.
        In addition to that, compute the contents for the message and program a review for later.
        """
        self.add_message_no_output(message)

        with DatabaseConnection() as session:
            print("Building context")
            context_input = CREATE_CONTEXT_PROMPT.invoke({"conversation": self.history.pretty()})

            print(context_input)

            context = Context.from_input(session, input=context_input.model_dump(exclude_none=True))
            contents = context.get_content(session)
            # Extract the cluster content using the special iterator on the contents object
            # Build the prompt and print it
            
            # import streamlit
            # streamlit.info([content.text for content in contents])

        if len(self.history) >= 3:
            self.pending_reviews.append(ReviewBuffer(self.history.last(3), {layer_name: accessor.id for layer_name, accessor in context.items()}))
        