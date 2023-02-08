FROM python:3.11.1

ADD  spam classifier ..

RUN pip install -r requirements.txt

