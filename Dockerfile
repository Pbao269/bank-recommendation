FROM python:3.11-slim

WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and application code
COPY *.pkl .
COPY main.py .

# Set environment variables
ENV PORT=8000
# API_KEY is already set with a default in the code
# You can override it during deployment if needed

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]