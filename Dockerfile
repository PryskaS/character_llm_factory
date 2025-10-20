# Stage 1: Use an official lightweight Python image as a parent image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the file with our dependencies first to leverage Docker's build cache
COPY requirements.txt .

# Install the dependencies. This will include torch and transformers.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project context into the container's working directory.
# This is the CRUCIAL step that brings our code AND our trained model artifact (`rick-llm-final`)
# into the Docker image.
COPY . .

# Tell Docker that the container will listen on port 8001
EXPOSE 8001

# The command to run our application when the container starts
# We specify the module path `serving.main` to our FastAPI app object
CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8001"]