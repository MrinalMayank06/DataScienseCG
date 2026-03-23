from pydantic import BaseModel


class AnalyticsSummaryResponse(BaseModel):
    dataset_path: str
    total_students: int
    average_score_by_course: dict[str, float]
    pass_rate_by_course: dict[str, float]
    top_performers: list[dict]
    insights: list[str]


class ChartPathsResponse(BaseModel):
    bar_chart_path: str
    line_chart_path: str
