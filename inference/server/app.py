import cv2
import numpy as np
import tritonclient.http as httpclient








from flask import Flask, request, jsonify
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    url = 'localhost:8000'
    model_name = 'classifier'
    model_version = ""
    triton_client = httpclient.InferenceServerClient(url=url)

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)  # Save file first
    
    # Read the image using OpenCV from disk
    image = cv2.imread(file_path)



    
    # image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    resized_image = cv2.resize(image, (224, 224))

    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB) / 255.0

    transposed_image = np.transpose(rgb_image, (2, 0, 1))

    input_data = np.expand_dims(transposed_image, axis=0).astype(np.float32)

    inputs = [httpclient.InferInput('actual_input', input_data.shape, 'FP32')]
    print("inputs:::::",inputs)
    inputs[0].set_data_from_numpy(input_data)
    print("inputs[0]:::::",inputs[0])

    response = triton_client.infer(model_name=model_name,
                                inputs=inputs,
                                model_version=model_version)



    output = response.as_numpy('output')  # Replace with actual output
    print(output)
    
    
    return jsonify({'message': 'File uploaded successfully', 'file_path': file_path,})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)







