FROM python:3.10-slim

# התקנת FontForge ו־libGL עבור OpenCV
RUN apt-get update && \
    apt-get install -y fontforge python3-dev build-essential libgl1 && \
    apt-get clean

# התקנת ספריות פייתון
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# העתקת הקוד
COPY . /app
WORKDIR /app

EXPOSE 10000

CMD ["python", "backend/server.py"]
