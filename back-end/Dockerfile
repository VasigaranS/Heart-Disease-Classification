# Dockerfile
FROM python:3.8

WORKDIR /app

COPY . .



RUN pip install --no-cache-dir -r back-end/requirements3.txt
#RUN pip install --no-cache-dir -r requirements3.txt

ENV FLASK_APP=flask_app

EXPOSE 5000


CMD ["python", "flask_app.py"]

