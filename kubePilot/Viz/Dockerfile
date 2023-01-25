FROM python:3.8-slim-buster

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]