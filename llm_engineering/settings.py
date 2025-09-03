from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LinkedIn Credentials
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None