FROM python:3.10-slim

ENV PYHTONUNBUFFERED=1
RUN apt-get update \
  && apt-get -y install tesseract-ocr \
  && apt-get -y install ffmpeg libsm6 libxext6

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -r requiriements.txt

COPY . .

CMD ["python", "client_main.py"]