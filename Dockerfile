FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "11434"]