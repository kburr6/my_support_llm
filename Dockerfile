# Dockerfile

# Step 1: Base image
FROM python:3.11-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy the model download script
COPY download_model.py .

# Step 5: Run the download script to bake the model into the image.
# This will now save the model to '/app/embedding_model'
RUN python download_model.py

# Step 6: Copy the rest of your application code
COPY . .

# Step 7: Expose the Streamlit port
EXPOSE 8501

# Step 8: Define the run command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]