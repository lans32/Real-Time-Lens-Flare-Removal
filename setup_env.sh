#!/bin/bash

# Проверяем наличие Python 3.10
if ! command -v python3.10 &> /dev/null; then
    echo "Python 3.10 не установлен. Установите его с помощью:"
    echo "brew install python@3.10"
    exit 1
fi

# Создаем виртуальное окружение с Python 3.10
python3.10 -m venv venv

# Активируем виртуальное окружение
source venv/bin/activate

# Обновляем pip
pip install --upgrade pip

# Устанавливаем зависимости
pip install -r requirements.txt

echo "Виртуальное окружение успешно создано и настроено!"
echo "Для активации окружения выполните: source venv/bin/activate" 