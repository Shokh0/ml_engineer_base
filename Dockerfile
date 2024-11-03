# Базовый образ. Для ML обычно подходит Python или образ с поддержкой GPU, например, от NVIDIA
FROM python:3.8-slim

# Устанавливаем системные зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libssl-dev \
        libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем необходимые зависимости через pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаем рабочую директорию в контейнере
WORKDIR /app

# Копируем проект в контейнер
COPY . /app

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Если используется GPU, можно добавить зависимости CUDA (требует базового образа NVIDIA)
# Например, для PyTorch:
# RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Запуск приложения (измените в зависимости от используемой библиотеки/фреймворка)
# Например, для FastAPI
CMD ["python", "app.py"]
