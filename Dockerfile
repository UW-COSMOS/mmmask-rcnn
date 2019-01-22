FROM anibali/pytorch:cuda-10.0

RUN pip install -r /app/requirements.txt

RUN python /app/setup.py build essential

