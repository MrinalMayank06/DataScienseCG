from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


class AnalyticsService:
    def __init__(self, dataset_path: Path = DATA_PATH):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.dataset_path)

    def generate_summary(self) -> dict:
        average_score_by_course = (
            self.df.groupby("course")["score"].mean().round(2).sort_values(ascending=False).to_dict()
        )
        pass_rate_by_course = (
            (self.df.groupby("course")["passed"].mean() * 100).round(2).sort_values(ascending=False).to_dict()
        )
        top_performers = (
            self.df.sort_values(by="score", ascending=False)
            .head(5)[["name", "course", "score", "attendance"]]
            .to_dict(orient="records")
        )
        strongest_course = max(average_score_by_course, key=average_score_by_course.get)
        highest_pass_course = max(pass_rate_by_course, key=pass_rate_by_course.get)
        insights = [
            f"{strongest_course} has the highest average score at {average_score_by_course[strongest_course]:.2f}.",
            f"{highest_pass_course} has the strongest pass rate at {pass_rate_by_course[highest_pass_course]:.2f}%.",
        ]
        return {
            "dataset_path": str(self.dataset_path),
            "total_students": int(len(self.df)),
            "average_score_by_course": average_score_by_course,
            "pass_rate_by_course": pass_rate_by_course,
            "top_performers": top_performers,
            "insights": insights,
        }

    def generate_charts(self) -> dict:
        avg_scores = self.df.groupby("course")["score"].mean().sort_values(ascending=False)
        attendance_trend = self.df.sort_values("student_id")[["student_id", "attendance"]]

        bar_path = ARTIFACTS_DIR / "average_score_by_course.png"
        line_path = ARTIFACTS_DIR / "attendance_trend.png"

        plt.figure(figsize=(8, 5))
        avg_scores.plot(kind="bar")
        plt.title("Average Score by Course")
        plt.xlabel("Course")
        plt.ylabel("Average Score")
        plt.tight_layout()
        plt.savefig(bar_path)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(attendance_trend["student_id"], attendance_trend["attendance"])
        plt.title("Attendance Trend by Student ID")
        plt.xlabel("Student ID")
        plt.ylabel("Attendance")
        plt.tight_layout()
        plt.savefig(line_path)
        plt.close()

        return {
            "bar_chart_path": str(bar_path),
            "line_chart_path": str(line_path),
        }
