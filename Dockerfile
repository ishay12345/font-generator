# בסיס: פייתון 3.10 רזה
FROM python:3.10-slim

# ודא שכל פלט של פייתון יוצא מיידית
ENV PYTHONUNBUFFERED=1

# התקנות מערכת: FontForge, Potrace, ועוד תלויות
RUN apt-get update && \
    apt-get install -y \
        fontforge \
        potrace \
        libgl1 \
        python3-dev \
        build-essential && \
    apt-get clean

# יצירת הפניות של STDOUT ו־STDERR עבור לוגים
RUN ln -sf /dev/stdout /var/log/fontforge.out && \
    ln -sf /dev/stderr /var/log/fontforge.err

# התקנת ספריות פייתון מתוך requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# העתקת כל קבצי הפרויקט
COPY . /app
WORKDIR /app

# פתיחת הפורט ש-Flask משתמש בו
EXPOSE 10000

# פקודת הריצה של השרת
CMD ["python", "backend/server.py"]
