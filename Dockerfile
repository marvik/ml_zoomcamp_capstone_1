FROM python:3.12-slim


RUN pip --no-cache-dir install pipenv


WORKDIR /app


COPY ["Pipfile", "Pipfile.lock", "./"]


RUN pipenv install --deploy --system && \
    rm -rf /root/.cache


COPY ["predict.py", "xgboost_model.bin", "./"]


EXPOSE 9696


ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]