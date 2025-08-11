# 1. Start from an official Python base image
FROM python:3.9

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# 4. Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Download the spaCy model during the build
RUN python -m spacy download en_core_web_sm

# 6. Copy the rest of your app's files into the container
# (This includes streamlit_app.py, the .joblib model files, etc.)
COPY . /code

# --- THE FIX IS HERE ---
# 7. Create the .streamlit directory and config file to disable telemetry
# This prevents the PermissionError when the app starts.
RUN mkdir -p /.streamlit
RUN echo '[browser]\ngatherUsageStats = false' > /.streamlit/config.toml

# 8. Define the command to run when the container starts
CMD ["streamlit", "run", "streamlit_app.py", "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false", "--server.fileWatcherType", "none"]