FROM python:3.9 as base

# Set workdir
WORKDIR /home/src

# Update pip
RUN pip3 install --upgrade pip

# Copy requirements
COPY ./requirements.txt .

# Install requirements
RUN pip install --no-cache-dir --upgrade -r /home/src/requirements.txt

# Set python path
ENV PYTHONPATH=$PYTHONPATH:/home/src/

# Copy info inside 
COPY . home/src/

# FROM base as test
# RUN ["pytest", "-v", "/src/tests"]

# Start the api service
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]