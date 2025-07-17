# בחר בסיס עם פייתון 3.10
FROM python:3.10-slim

# התקן כלים בסיסיים
RUN apt-get update && apt-get install -y \
    build-essential \
    potrace \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean

# הגדר תיקיית עבודה
WORKDIR /app

# העתק את כל קבצי הפרויקט
COPY . .

# התקן את התלויות
RUN pip install --upgrade pip
RUN pip install flask flask-cors opencv-python numpy Pillow ufoLib2 fonttools cu2qu defcon ufo2ft

# הגדר את הפורט
EXPOSE 10000

# הפעל את השרת
CMD ["python", "backend/server.py"]
