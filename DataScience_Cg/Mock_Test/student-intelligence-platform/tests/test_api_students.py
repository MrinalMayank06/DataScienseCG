def test_create_and_get_student(client):
    payload = {"name": "Kiki", "age": 22, "course": "AI"}
    create_response = client.post("/api/v1/students", json=payload)
    assert create_response.status_code == 201
    created = create_response.json()["data"]
    assert created["name"] == "Kiki"

    get_response = client.get(f"/api/v1/students/{created['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["data"]["course"] == "AI"


def test_filter_students_by_course(client):
    client.post("/api/v1/students", json={"name": "Anu", "age": 22, "course": "AI"})
    client.post("/api/v1/students", json={"name": "Ria", "age": 23, "course": "Cloud"})

    response = client.get("/api/v1/students?course=AI")
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 1
    assert data[0]["course"] == "AI"


def test_delete_student(client):
    create_response = client.post("/api/v1/students", json={"name": "Mia", "age": 21, "course": "Web Dev"})
    student_id = create_response.json()["data"]["id"]

    delete_response = client.delete(f"/api/v1/students/{student_id}")
    assert delete_response.status_code == 200

    get_response = client.get(f"/api/v1/students/{student_id}")
    assert get_response.status_code == 404
