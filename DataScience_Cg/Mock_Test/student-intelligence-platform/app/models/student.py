from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    course: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
