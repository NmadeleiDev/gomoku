FROM python:3.10.2

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt --disable-pip-version-check

COPY . /app

CMD python main.py -p1 AI -p2 AI
