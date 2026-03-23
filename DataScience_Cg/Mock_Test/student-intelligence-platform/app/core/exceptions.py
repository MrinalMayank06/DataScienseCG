from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class NotFoundError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class ConflictError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def not_found_handler(_: Request, exc: NotFoundError):
        return JSONResponse(
            status_code=404,
            content={"message": exc.detail, "data": None},
        )

    @app.exception_handler(ConflictError)
    async def conflict_handler(_: Request, exc: ConflictError):
        return JSONResponse(
            status_code=409,
            content={"message": exc.detail, "data": None},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"message": "Validation error", "errors": exc.errors(), "data": None},
        )

    @app.exception_handler(Exception)
    async def generic_handler(_: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error", "detail": str(exc), "data": None},
        )
