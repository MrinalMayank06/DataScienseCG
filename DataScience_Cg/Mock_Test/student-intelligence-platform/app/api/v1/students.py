from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.student import APIResponse, StudentCreate, StudentRead, StudentUpdate
from app.services.student_service import StudentService

router = APIRouter(prefix="/students")


@router.get("", response_model=APIResponse[list[StudentRead]])
async def list_students(
    course: Optional[str] = Query(default=None),
    name_query: Optional[str] = Query(default=None, description="Partial name search"),
    db: Session = Depends(get_db),
):
    students = StudentService(db).list_students(course=course, name_query=name_query)
    return APIResponse(message="Students fetched successfully", data=students)


@router.post("", response_model=APIResponse[StudentRead], status_code=status.HTTP_201_CREATED)
async def create_student(payload: StudentCreate, db: Session = Depends(get_db)):
    student = StudentService(db).create_student(payload)
    return APIResponse(message="Student created successfully", data=student)


@router.get("/{student_id}", response_model=APIResponse[StudentRead])
async def get_student(student_id: int, db: Session = Depends(get_db)):
    student = StudentService(db).get_student(student_id)
    return APIResponse(message="Student fetched successfully", data=student)


@router.put("/{student_id}", response_model=APIResponse[StudentRead])
async def update_student(student_id: int, payload: StudentUpdate, db: Session = Depends(get_db)):
    student = StudentService(db).update_student(student_id, payload)
    return APIResponse(message="Student updated successfully", data=student)


@router.delete("/{student_id}", response_model=APIResponse[dict])
async def delete_student(student_id: int, db: Session = Depends(get_db)):
    StudentService(db).delete_student(student_id)
    return APIResponse(message="Student deleted successfully", data={"id": student_id})
