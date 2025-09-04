from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LinkedIn Credentials
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None

    # MongoDB Configuration
    DATABASE_NAME: str = "llm_twin"
    DATABASE_HOST: str = "mongodb://localhost:27017"