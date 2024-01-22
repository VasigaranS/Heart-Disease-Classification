# Dockerfile
FROM python:3.8

WORKDIR /app

COPY . .



RUN pip install --no-cache-dir -r requirements3.txt

ENV FLASK_APP=flask_app

EXPOSE 5000

# Define environment variable
ENV AWS_ACCESS_KEY='AKIAQOG46ZJRFTR27GAW'
ENV AWS_SECRET_KEY='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'
ENV S3_BUCKET='heartdataset'

CMD ["python", "flask_app.py"]

