from pydantic import BaseModel


class ProbabilityResponse(BaseModel):
    probability: float
    percentage: float
    explanation: str
