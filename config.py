from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file


class Settings(BaseSettings):
    OPEN_AI_TOKEN: str | None = os.getenv("OPEN_AI_TOKEN")


settings = Settings()
