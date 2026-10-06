FROM python:3.12-slim

WORKDIR /app


COPY requirements-rag.txt .

RUN pip install --no-cache-dir -r requirements-rag.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
