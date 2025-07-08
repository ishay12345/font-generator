# בסיס: פייתון 3.10
FROM python:3.10-slim

# התקנת FontForge ותלויות בסיסיות
RUN apt-get update && \
    apt-get install -y fontforge python3-dev build-essential && \
    apt-get clean

# התקנת ספריות פייתון
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# העתקת כל קבצי הפרויקט
COPY . /app
WORKDIR /app

# פתיחת פורט ש-Flask משתמש בו
EXPOSE 10000

# הרצת השרת שלך
CMD ["python", "backend/server.py"]
