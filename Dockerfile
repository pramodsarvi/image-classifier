 FROM nvcr.io/nvidia/pytorch:25.01-py3-igpu
 COPY requirements.txt /workspace/requirements.txt
 RUN pip install -r /workspace/requirements.txt
 COPY . /workspace
 WORKDIR /workspace
 #CMD ["python", "main.py"]