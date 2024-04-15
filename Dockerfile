# Use a Python base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN pip install -r requirements.txt

# Copy local contents into the container at /app
COPY app.py .
COPY models/ models/
COPY static/ static/

# Expose port 8501 for Streamlit
EXPOSE 8501

# Specify the command to run when the container starts
CMD ["streamlit", "run", "app.py"]