from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field, ConfigDict

DataT = TypeVar("DataT")


class APIResponse(BaseModel, Generic[DataT]):
    message: str
    data: Optional[DataT] = None


class StudentBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    age: int = Field(..., ge=16, le=100)
    course: str = Field(..., min_length=2, max_length=80)


class StudentCreate(StudentBase):
    pass


class StudentUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=120)
    age: Optional[int] = Field(default=None, ge=16, le=100)
    course: Optional[str] = Field(default=None, min_length=2, max_length=80)


class StudentRead(StudentBase):
    id: int
    model_config = ConfigDict(from_attributes=True)
