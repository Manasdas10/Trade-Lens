# Use official slim Python image
FROM python:3.11-slim

# Install basic system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set standard working directory
WORKDIR /app

# Copy requirements and install python packages
COPY ai-python/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose ports: 8000 for FastAPI, 8501 for Streamlit Dashboard
EXPOSE 8000
EXPOSE 8501

# Create startup script to run both services simultaneously
RUN echo '#!/bin/bash\n\
cd /app/ai-python\n\
python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8000 &\n\
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Execute the startup script
CMD ["/app/start.sh"]
