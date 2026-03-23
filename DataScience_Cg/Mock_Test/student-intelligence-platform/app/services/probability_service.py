class ProbabilityService:
    def calculate_pass_probability(self, passed_count: int, total_count: int) -> dict:
        probability = passed_count / total_count
        percentage = round(probability * 100, 2)
        explanation = (
            f"Out of {total_count} students, {passed_count} passed. "
            f"The probability of passing is {probability:.2f}, which means {percentage}% of the dataset passed."
        )
        return {
            "probability": round(probability, 4),
            "percentage": percentage,
            "explanation": explanation,
        }
