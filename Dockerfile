# Use NVIDIA PyTorch container as the base image
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Set up the working directory
WORKDIR /workspace

# Copy requirements file and install dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

EXPOSE 8080

ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080"]
