FROM python:3.10-slim

# התקנות מערכת: fontforge, potrace, ועוד
RUN apt-get update && \
    apt-get install -y \
        fontforge \
        potrace \
        libgl1 \
        python3-dev \
        build-essential && \
    apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 10000

CMD ["python", "backend/server.py"]
