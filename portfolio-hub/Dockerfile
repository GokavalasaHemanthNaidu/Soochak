FROM python:3.11-slim

WORKDIR /app

RUN pip install streamlit

COPY hub.py .

EXPOSE 8501

CMD ["streamlit", "run", "hub.py", "--server.port=8501", "--server.address=0.0.0.0"]
