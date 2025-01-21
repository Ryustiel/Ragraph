
import os

from typing import (
    List,
    Optional,
)

from openai import AzureOpenAI

class Vectorizer:
    """
    Handles turning a text input into a vector (currently using the remote AzureOpenAI service).
    """
    global_client: Optional[AzureOpenAI] = None

    def __init__(self, 
        model: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
    ):
        if (
            api_key is None 
            and api_version is None 
            and azure_endpoint is None
        ):
            self.use_global_client = True
        else:
            self.use_global_client = False

        self._client = None

        self.model = model  # "text-embedding-3-large" by default, can be changed in the constructor.
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        
    @property
    def client(self) -> AzureOpenAI:
        if self._client is None:
            if self.use_global_client:  # Build the local client from the global client
                self._client = Vectorizer.get_global_client()
            else:
                self._client = self.create_azure_client(
                    api_key=self.api_key,
                    api_version=self.api_version,
                    azure_endpoint=self.azure_endpoint,
                )
        return self._client
    
    @staticmethod
    def create_azure_client(
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
    ) -> AzureOpenAI:
        """Create an AzureOpenAI instance, use the default variables if available and handle errors if anything is missing."""
        if api_key is None:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if api_version is None:
            api_version = os.getenv("OPENAI_API_VERSION")
        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if api_key is None:
            raise ValueError("AZURE_OPENAI_API_KEY is missing from the environment.")
        if api_version is None:
            raise ValueError("AZURE_OPENAI_API_VERSION is missing from the environment.")
        if azure_endpoint is None:
            raise ValueError("AZURE_OPENAI_ENDPOINT is missing from the environment.")

        return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint)

    @classmethod
    def get_global_client(cls) -> AzureOpenAI:
        """Return the global client instance. Create a new AzureOpenAI client if no global client is yet defined."""
        if cls.global_client is None:
            cls.global_client = cls.create_azure_client()
        return cls.global_client
    
    @classmethod
    def process(cls, text: str, model: str = "text-embedding-3-large") -> List[float]:
        """
        Vectorize the input vector using the global, class wide client. 
        Size = 3072 if model == "text-embedding-3-large"

        NOTE : This is identical to Vectorizer().vectorize(text)
        """
        client = cls.get_global_client()
        response = client.embeddings.create(
            input = text,
            model = model
        )
        return response.data[0].embedding

    def vectorize(self, text: str) -> List[float]:
        """
        Vectorize the input vector. 
        Size = 3072 if model == "text-embedding-3-large"
        """
        response = self.client.embeddings.create(
            input = text,
            model = self.model
        )
        return response.data[0].embedding
    
    def __call__(self, text: str) -> List[float]:
        """
        Vectorize the input vector. 
        Size = 3072 if model == "text-embedding-3-large"
        """
        return self.vectorize(text)
