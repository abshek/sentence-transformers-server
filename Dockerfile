# Use the official Python image as the base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Install any necessary dependencies
RUN apt-get update && apt-get install -y git
RUN pip install flask
RUN pip install sentence-transformers

# Copy the rest of the application code into the container
RUN git clone https://github.com/abshek/sentence-transformers-server.git .

# Expose the desired port
EXPOSE 8081

# Command to run the Python script
CMD ["python", "embedding_server.py"]