FROM tiangolo/uvicorn-gunicorn-fastapi

COPY app/requirements.txt .
RUN pip install -r requirements.txt

COPY ./app .

EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]