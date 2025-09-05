FROM python:3.10-slim

# Set the working directory in docker
WORKDIR /pokemon

# Install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  git && \
  # Clean up apt cache to reduce image size.
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Copy source-code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r baselines/requirements.txt

# Command to run at container start
CMD [ "python", "baselines/run_baseline_parallel.py" ]
