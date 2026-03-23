from app.services.probability_service import ProbabilityService


def test_pass_probability():
    result = ProbabilityService().calculate_pass_probability(30, 100)
    assert result["probability"] == 0.3
    assert result["percentage"] == 30.0
    assert "30 passed" in result["explanation"]
