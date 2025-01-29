
from typing import (
    List,
    Dict,
    Callable,
)

from ..database import DatabaseConnection, Content, Context
from ._llm import LLM

from raphlib import ChatHistory, ChatMessage, LLMFunction
from datetime import datetime, timedelta


CHUNK_REVIEW = LLMFunction(LLM, 
                            "In the conversation, was the provided chunk of information useful to the AI in any way? \n\nCHUNK: {chunk} \n\nCONVERSATION: {conversation}",
                            
                            answer = bool
                )
MISSING_CONTENT_REVIEW = LLMFunction(LLM, 
                            """
                            You will be provided a conversation between an AI and users.
                            The AI was "remembering" information and instructions in the middle of the conversation.

                            Did the AI miss some important information or instructions that made it fail to reply properly? 
                            If yes, what was it that the AI had to know in order to respond properly, but obviously did not know?

                            If any, write it down as chunks of information in your structured output, as if to be inserted into a vector store,
                            so that they can be remembered if that context occurs again.
                            For example in your output : ["User likes mangoes"]

                            \n\nCONVERSATION: {conversation}
                            \n\nAI REMEMBERED IN THE MIDDLE OF THE CONVERSATION: {content}
                            """,

                            what_they_had_to_know = ["example", ...]
                )
UESLESS_CONTENT_REVIEW = LLMFunction(LLM,
                            """
                            You will be provided a conversation between an AI and users.
                            The AI was "remembering" information and instructions in the middle of the conversation.

                            Did any of this information and instructions were misleading / should not be thought of in this context?
                            If yes, write them down in your structured output, as if to be queried from a vector store so that they can be remembered less often in that context.

                            NOTE : The data you were provided with do not include what the AI remembers in its **last message**, 
                            your answer should only rely on the relationship between it's "middle messages" and how it decided to answer afterwards.

                            \n\nCONVERSATION: {conversation}
                            \n\nAI REMEMBERED IN THE MIDDLE OF THE CONVERSATION: {content}
                            """,

                            mislaeding_chunks = ["example", ...]
                        )

HAS_USER_INSTRUCTION = LLMFunction(LLM,
                            """
                            The AI talked in the middle of the conversation.
                            Are the user(s) providing information or commenting on the AI's behavior? (should forget, should do or not do, should know that, ...)
                            Respond with True if Yes, False if No.
                            The answer is also False if only the AI provided information and the users did not comment on the information or the AI behavior.
                            """,

                            answer = bool
                        )

# When to make reviews depending on the number of past reviews and the characteristics of the input message and surrounding user messages
REVIEW_CONDITION: Callable[[int, ChatHistory], bool] = lambda x, y: x > 1 and len(y) > 2
# TODO : Only review if the "weight" of the conversation is centered around the middle AI message itself or user messages directly following it.
# TODO : Always review if user is asking ai to remember or forget information (make a prompt for that)



class ReviewBuffer:
    """
    An instance of this class typically stores a triplet
    (a few messages prior to the target; the target message; a context input)
    """
    REVIEW_EXPIRATION_TIME = timedelta(minutes=30)
    
    def __init__(self, backwards: ChatHistory, context: Dict[str, int]):
        """
        Backwards contains the ChatHistory of a couple messages before the latest message.
        It also contains the latest human message - the context has been computed for.

        Context represents the context from which contents have been pulled for the rag.
        It's a mapping from layer_name to accessor_node.
        
        Upon producing the review, the context will be used to fetch content nodes and their weights.
        A LLMFunction will be used to determine in 2 ways how to update the weights of the content nodes.
        1. Random scoring on the highest ranked contents.
        2. Spontaneous remarks on superfluous or missing pieces of content, that will trigger stronger weight updates or content node creation.

        The production of the review will be triggered by an input ChatHistory which will be compared against the latest message in self.backwards
        If the input ChatHistory contains the next part of the discussion after the latest message in here, 
        then we'll have enough context to produce a review.

        This object will be deleted after the review has been issued and the weight have been updated.
        """
        self.history = backwards
        self.context = context
        self.completed = False
        self.first_message_time: datetime = datetime.now()

    def add_message(self, message: ChatMessage):
        """
        Add a new message to the history.
        If the message count is enough, then perform the review and mark itself as completed.
        """
        if datetime.now() - self.first_message_time > self.REVIEW_EXPIRATION_TIME:  # Skips the review if no message had been received for a long time.
            self.completed = True
            return

        self.history.messages.append(message)

        # Populate the prompt of the review and show what's in it on streamlit
        
        if len(self.history) > 7:
            self.perform_review()

    def perform_review(self):
        """
        Perform the review on the history and context.
        """
        # TODO: Implement the review logic here
        with DatabaseConnection() as session:
            context = Context.from_input(session, self.context)
            contents = context.get_content(session)

            print("Performing review with", contents, self.context)

            missing_content = MISSING_CONTENT_REVIEW.invoke({"conversation": self.history.pretty(), "content": str(contents)})
            print(missing_content)

            for cnt in missing_content:
                context.add_content(session, cnt)

            # Pick (not totally) randomly chunks from the contents
            # Perform the review on the chunks and update the weights in the graph

        self.completed = True
        
    def is_complete(self) -> bool:
        """Check if the review has been performed."""
        return self.completed
    