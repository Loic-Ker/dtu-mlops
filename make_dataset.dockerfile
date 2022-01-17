# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/




RUN pip install --default-timeout=100 -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/data/make_dataset.py"]