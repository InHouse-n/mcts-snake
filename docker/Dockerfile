#syntax=docker/dockerfile:1

FROM python:3
WORKDIR /app
COPY . /app
RUN pip install "poetry==1.3.2"
RUN poetry config virtualenvs.create false
RUN poetry install --only main

CMD [ "make", "run" ]