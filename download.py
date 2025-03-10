import os
import logging
import shutil
import subprocess
import time
from pathlib import Path
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuration for Triton model."""
    def __init__(self, name: str):
        self.name = name
        self.platform = "onnxruntime_onnx"
        self.max_batch_size = 32
        self.input_dtype = "TYPE_FP32"
        self.input_dims = [1,3, 224, 224]        
        self.output_dtype = "TYPE_FP32"
        self.output_dims = [-1,2]

def download_mlflow_model(mlflow_url: str, model_name: str, model_version: int, output_path: str,runs_id: str,model: str) -> str:
    """Downloads an MLflow model for Triton deployment."""
    try:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(output_dir, 0o755)

        logger.info(f"Setting MLflow tracking URI: {mlflow_url}")
        mlflow.set_tracking_uri(mlflow_url)

        # model_uri = f"models:/{model_name}/{model_version}"
        #  = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=output_path))

        model_uri = f'runs:/{runs_id}/{model}'  # Change this to your MLflow model path
        onnx_model = mlflow.onnx.load_model(model_uri)
        import onnx 
        # Save ONNX model as a standalone file (optional)
        
        onnx.save(onnx_model, os.path.join(output_path , "model.onnx"))

        model_dir = output_dir / model_name
        version_dir = model_dir / str(model_version)
        version_dir.mkdir(parents=True, exist_ok=True)

        source_model_path = Path(os.path.join(output_path , "model.onnx"))
        dest_model_path = Path(os.path.join(version_dir , "model.onnx"))

        if source_model_path.exists():
            shutil.copy(source_model_path, dest_model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {source_model_path}")

        model_config = ModelConfig(name=model_name)
        config_text = f"""
        name: "{model_config.name}"
        platform: "{model_config.platform}"
        max_batch_size: {model_config.max_batch_size}
        input: [    
        {{
            "name": "actual_input", 
            "data_type": "{model_config.input_dtype}", 
            "dims": {model_config.input_dims}
        }}
        ]
        output: [ {{
            "name": "output",
            "data_type": "{model_config.output_dtype}",
            "dims": {model_config.output_dims}
        }}]
        """
        config_path = model_dir / "config.pbtxt"
        # config_path.write_text(config_text.strip())

        logger.info("Model downloaded and prepared successfully.")
        return str(model_dir)

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        raise
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"MLflow error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def deploy_triton_model(model_name: str, model_path: str):
    """Deploy a Triton model."""
    docker_image = "nvcr.io/nvidia/tritonserver:23.04-py3"

    try:
        # Check if Docker is running
        if subprocess.run("docker info", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            raise RuntimeError("Docker daemon is not running. Please start Docker.")

        # Stop and remove existing Triton container
        subprocess.run("docker rm -f triton_server", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        logger.info("Removed existing Triton container.")

        # Run Triton server
        subprocess.run(
            f"docker run --gpus all --rm -p 8000:8000 -p 8001:8001 "
            f"-v {model_path}:/models/{model_name} {docker_image} tritonserver --model-repository=/models",
            shell=True,
            check=True
        )
        logger.info(f"Triton server started with model: {model_name}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Docker command failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    mlflow_url = "http://13.127.65.67:5000"
    model_name = "triton_test_model_1"
    model_version = 2
    output_path = "/tmp/models"
    runs_id = "e694558ad8bb47efbd636141a23b016f"
    model = "model_onnx_best"

    try:
        model_path = download_mlflow_model(mlflow_url, model_name, model_version, output_path, runs_id,model)
        deploy_triton_model(model_name, model_path)
    except Exception as e:
        logger.error(f"Script failed: {e}")