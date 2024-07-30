# Base image
FROM amd64/python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir wandb

# Command to run the server
CMD ["wandb", "server", "--port", "8080"]