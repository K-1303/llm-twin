from pydantic import BaseModel
from abc import ABC
from typing import TypeVar, Generic

T = TypeVar("T", bound="NoSQLBaseDocument")

class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    pass