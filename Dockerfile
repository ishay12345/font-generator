# בסיס: פייתון 3.10 (גרסה רזה)
FROM python:3.10-slim

# התקנת FontForge, potrace, ותלויות מערכת חיוניות ל־OpenCV ו־Pillow
RUN apt-get update && \
    apt-get install -y fontforge potrace libgl1 python3-dev build-essential && \
    apt-get clean

# התקנת ספריות פייתון מהקובץ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# העתקת כל קבצי הפרויקט
COPY . /app
WORKDIR /app

# פתיחת פורט ש-Flask מאזין עליו
EXPOSE 10000

# פקודת ההרצה של השרת
CMD ["python", "backend/server.py"]
