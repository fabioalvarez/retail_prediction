FROM python:3.9 as base

# Update pip
RUN pip3 install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install requirements
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Set python path
ENV PYTHONPATH=$PYTHONPATH:/home/src/

# Set workdir
WORKDIR /home/src

# Copy info inside 
ADD . /home/src/api/

# Start the api service
FROM base as build
CMD ["python3", "/home/src/api/main.py"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]