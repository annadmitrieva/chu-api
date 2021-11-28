FROM python:3.7
COPY requirements.txt .
RUN pip install --default-timeout=100 -r requirements.txt

COPY ./api /api/api

ENV PYTHONPATH=/api
WORKDIR /api


EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["api.main:app", "--host", "0.0.0.0"]