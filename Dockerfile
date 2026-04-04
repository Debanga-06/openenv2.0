FROM python:3.13-slim

WORKDIR /app

# First copy the requirements file
COPY server/requirements.txt ./server/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy the rest of the application
COPY . .

# Change working directory to server where app.py and environment.py live
WORKDIR /app/server

# Expose port required by Hugging Face Spaces
EXPOSE 7860

# Command to run the Fastapi application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
