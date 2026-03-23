from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.models.student import Student
from app.schemas.student import StudentCreate, StudentUpdate


class StudentService:
    def __init__(self, db: Session):
        self.db = db

    def list_students(self, course: Optional[str] = None, name_query: Optional[str] = None):
        stmt = select(Student)
        if course:
            stmt = stmt.where(Student.course.ilike(course))
        if name_query:
            stmt = stmt.where(Student.name.ilike(f"%{name_query}%"))
        stmt = stmt.order_by(Student.id.asc())
        return self.db.execute(stmt).scalars().all()

    def create_student(self, payload: StudentCreate) -> Student:
        student = Student(**payload.model_dump())
        self.db.add(student)
        self.db.commit()
        self.db.refresh(student)
        return student

    def get_student(self, student_id: int) -> Student:
        student = self.db.get(Student, student_id)
        if not student:
            raise NotFoundError(f"Student with id={student_id} not found")
        return student

    def update_student(self, student_id: int, payload: StudentUpdate) -> Student:
        student = self.get_student(student_id)
        updates = payload.model_dump(exclude_none=True)
        for key, value in updates.items():
            setattr(student, key, value)
        self.db.commit()
        self.db.refresh(student)
        return student

    def delete_student(self, student_id: int) -> None:
        student = self.get_student(student_id)
        self.db.delete(student)
        self.db.commit()
