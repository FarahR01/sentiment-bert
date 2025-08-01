# Use official PyTorch image with CUDA 11.8
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
# Set working directory
WORKDIR /app

# Copy environment files
COPY requirements.txt ./
COPY environment.yml ./

# Install dependencies
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY ./app ./app

# Expose port for FastAPI
EXPOSE 8000

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]