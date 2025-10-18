from loguru import logger
from pydantic_settings import BaseSettings
from zenml.client import Client

class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    # LinkedIn Credentials
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None

    # MongoDB Configuration
    DATABASE_NAME: str = "llm-twin"
    DATABASE_HOST: str = ""

    # RAG
    TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"
    TEMPERATURE_INFERENCE: float = 0.0
    MAX_NEW_TOKENS_INFERENCE: int = 256
    TOP_P_INFERENCE: float = 0.9
    TEMPERATURE_INFERENCE: float = 0.0
    
    # QdrantDB Vector DB
    USE_QDRANT_CLOUD: bool = False
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_CLOUD_URL: str = "str"
    QDRANT_APIKEY: str | None = None

    # HuggingFace Configuration
    HF_TOKEN: str | None = None
    HF_USERNAME: str | None = None
    HF_MODEL_ID: str | None = None

    # Google Gemini Configuration
    GOOGLE_API_KEY: str | None = None
    GOOGLE_GEMINI_MODEL: str = "gemini-2.0-flash"

    # Opik / Comet ML Configuration
    COMET_API_KEY: str | None = None
    COMET_PROJECT: str | None = None

    # AWS Configuration
    AWS_REGION: str = "us-east-1"
    SAGEMAKER_ENDPOINT_INFERENCE: str = "llm-twin-inference-endpoint"


    # X ScrapingDog Configuration
    SCRAPINGDOG_API_KEY: str | None = None

    @classmethod
    def load_settings(cls) -> "Settings":
        """
        Tries to load the settings from the ZenML secret store. If the secret does not exist, it initializes the settings from the .env file and default values.

        Returns:
            Settings: The initialized settings object.
        """

        try:
            logger.info("Loading settings from the ZenML secret store.")

            settings_secrets = Client().get_secret("settings")
            settings = Settings(**settings_secrets.secret_values)
        except (RuntimeError, KeyError):
            logger.warning(
                "Failed to load settings from the ZenML secret store. Defaulting to loading the settings from the '.env' file."
            )
            settings = Settings()

        return settings

settings = Settings.load_settings()