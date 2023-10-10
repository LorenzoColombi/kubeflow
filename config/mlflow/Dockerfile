FROM python:3.7-slim-buster

#Aggiungere le dipendenze del database che si vuole usare
RUN pip3 install --upgrade pip && \
    pip3 install mlflow psycopg2-binary


ENTRYPOINT ["mlflow"," server"]