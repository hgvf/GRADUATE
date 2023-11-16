FROM python:3.8

RUN mkdir /GRADUATE
COPY . /GRADUATE
WORKDIR /GRADUATE

RUN \
apt-get update -y && \
apt-get install python3-pip -y && \
pip install -r requirements.txt && \
cd seisbench && \
pip install . 

CMD [ "python3" ]

