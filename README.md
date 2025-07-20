Quick Start: Deployment with Docker Compose

This guide will get the entire AI Support Assistant application running on your system.

Prerequisites:
Docker Desktop installed and running.
An internet connection for the initial setup.

Step 1: Launch the Application
Navigate to the project's root directory in your terminal and run the following command. This will build the application image, download the necessary service images, and start all the containers.

docker-compose up --build

The first time you run this, it will take several minutes to download the Python dependencies and the embedding model. You will see a lot of log output from the different services. Wait for the logs to slow down and for the app and db services to show that they are running and healthy.

Step 2: Pull the Language Model
The application needs a Large Language Model (LLM) to function. You must pull this model into the running Ollama service.
While the application is running (from Step 1), open a new, separate terminal window, navigate to the same project directory, and run the following command:

docker-compose exec ollama ollama pull llama3:8b

You will see a progress bar as the llama3:8b model is downloaded. This can take several minutes depending on your internet connection.

That's it!

Once the model pull is complete, your application is fully configured. You can now access the user interface by opening a web browser and navigating to:

http://localhost:8501

The database and the downloaded LLM are stored in persistent Docker volumes, so you will only need to perform Step 2 the very first time you set up the project. For all subsequent starts, you can simply run docker-compose up.