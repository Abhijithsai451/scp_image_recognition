
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project to the container
COPY . .

# Command to run the train script
CMD ["python","train.py","--config","config.json"]


# docker build -t scp_container
# docker run --gpus all scp_container