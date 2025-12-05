FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
ENV PORT=11434
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT}"