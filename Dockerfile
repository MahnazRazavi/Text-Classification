# Step 1: Use an official TensorFlow image as the base
FROM tensorflow/tensorflow:2.12.0

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file to the container
COPY requirements.txt .

# Step 4: Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the application code to the container
COPY . .

# Step 6: Set environment variables (if needed)
# Example: Setting a model path environment variable
ENV MODEL_INPUT_PATH="output/model/tcl"

# Step 7: Specify the command to run the application
CMD ["python", "test.py"]
