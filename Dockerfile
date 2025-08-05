FROM python:3.10

WORKDIR /app

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

EXPOSE 8000

# Use shell form CMD to allow $PORT environment variable expansion
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
