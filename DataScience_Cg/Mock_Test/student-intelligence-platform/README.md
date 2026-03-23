# Student Intelligence Platform

Production-style, modular, FastAPI-based intelligent system for student management, analytics, visualization, probability, and machine learning.

> Built from the requirements in the uploaded mock test document. fileciteturn0file0

## Architecture

```text
student_intelligence_platform/
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── schemas/
│   └── services/
├── ml/
├── data/
├── artifacts/
├── tests/
├── docker/
└── .github/workflows/
```

## Core features

- FastAPI REST API with async endpoints
- SQLAlchemy ORM with SQLite by default and PostgreSQL-ready config
- CSV analytics with Pandas
- Probability engine for pass-rate explanations
- Scikit-learn ML pipeline with persistence
- Charts generated via Matplotlib
- Pytest suite with API + unit tests
- Dockerized runtime
- GitHub Actions CI on push and pull request
- Structured logging and centralized exception handling

## Quick start in VS Code

### 1) Open the folder
- Extract the zip
- Open the `student_intelligence_platform` folder in VS Code

### 2) Create and activate virtual environment

#### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Windows CMD
```cmd
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Create env file
```bash
copy .env.example .env
```

On macOS/Linux:
```bash
cp .env.example .env
```

### 5) Run the API
```bash
uvicorn app.main:app --reload
```

Docs:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## API endpoints

### Student CRUD
- `GET /api/v1/students`
- `POST /api/v1/students`
- `GET /api/v1/students/{student_id}`
- `PUT /api/v1/students/{student_id}`
- `DELETE /api/v1/students/{student_id}`

### Filters
- `GET /api/v1/students?course=AI`
- `GET /api/v1/students?name_query=an`

### Analytics
- `GET /api/v1/analytics/summary`
- `GET /api/v1/analytics/charts`

### Probability
- `GET /api/v1/probability/pass?passed_count=30&total_count=100`

### ML
- `POST /api/v1/ml/train`
- `POST /api/v1/ml/predict`

## Sample student create request

```json
{
  "name": "Kiki Sharma",
  "age": 22,
  "course": "AI"
}
```

## Example curl commands

### Add student
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/students" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"Kiki Sharma\",\"age\":22,\"course\":\"AI\"}"
```

### Get all students
```bash
curl "http://127.0.0.1:8000/api/v1/students"
```

### Filter by course
```bash
curl "http://127.0.0.1:8000/api/v1/students?course=AI"
```

## Dataset used

`data/student_performance.csv`

Columns:
- student_id
- name
- age
- course
- hours_studied
- attendance
- assignments_completed
- score
- passed

## Business insights generated

The analytics module computes:
- average score by course
- pass rate by course
- top performers
- highest attendance groups
- sorted and filtered performance views

## Train ML model

```bash
python -m ml.train
```

This saves model to:
- `artifacts/student_model.joblib`

## Run tests

```bash
pytest -q
```

## Docker

Build:
```bash
docker build -f docker/Dockerfile -t student-intelligence-platform .
```

Run:
```bash
docker run --rm -p 8000:8000 --env-file .env student-intelligence-platform
```

## Docker vs virtual environment

### Virtual environment
- isolates Python packages only
- shares host OS and system libraries
- best for local development speed

### Docker
- packages app + runtime + dependencies together
- consistent across laptop, server, and CI
- stronger deployment parity and easier scaling

## Git workflow
```bash
git init
git add .
git commit -m "feat: initial student intelligence platform"
git checkout -b feature/api
git remote add origin <your-github-repo-url>
git push -u origin feature/api
```

## Postman tests
A starter collection is included:
- `postman/Student-Intelligence-Platform.postman_collection.json`
