FROM ubuntu:latest

USER root  

RUN sed -i 's/stretch/buster/g' /etc/apt/sources.list

# Cài đặt auditbeat
RUN apt-get update && \
    apt-get install -y --allow-unauthenticated wget && \
    wget https://artifacts.elastic.co/downloads/beats/auditbeat/auditbeat-8.15.0-amd64.deb && \
    dpkg -i auditbeat-8.15.0-amd64.deb && \
    apt-get clean && \
    rm auditbeat-8.15.0-amd64.deb

# Thiết lập auditbeat
COPY auditbeat.yml /usr/share/auditbeat/auditbeat.yml

# Quay lại user mặc định (nếu cần)
#USER www-data

# Khởi động auditbeat
CMD ["auditbeat", "-strict.perms=false", "-c", "/usr/share/auditbeat/auditbeat.yml"]

