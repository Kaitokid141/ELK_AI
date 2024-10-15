# Sử dụng image Python cơ bản
FROM python:3.6.7

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép các tệp cần thiết vào image
COPY requirements.txt .

# Cài đặt các thư viện
#RUN apt-get install -y build-essential
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng vào image
COPY app.py .
COPY model ./model
COPY tokenizer ./tokenizer

# Chỉ định cổng mà ứng dụng sẽ chạy
EXPOSE 5000

# Chạy ứng dụng
CMD ["python", "app.py"]
