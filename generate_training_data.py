import numpy as np
import cv2
import os
from tqdm import tqdm

def create_flare(image):
    # Создаем случайный блик
    height, width = image.shape[:2]
    flare = np.zeros_like(image)
    
    # Создаем несколько случайных точек для бликов
    num_flares = np.random.randint(3, 8)
    for _ in range(num_flares):
        # Случайная позиция
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        
        # Случайный размер
        size = np.random.randint(20, 100)
        
        # Случайный цвет (ближе к белому)
        color = np.random.randint(200, 255, 3)
        
        # Создаем круглый блик
        cv2.circle(flare, (x, y), size, color.tolist(), -1)
        
        # Добавляем размытие
        flare = cv2.GaussianBlur(flare, (0, 0), np.random.uniform(5, 15))
    
    # Смешиваем блик с изображением
    alpha = np.random.uniform(0.3, 0.7)
    result = cv2.addWeighted(image, 1, flare, alpha, 0)
    
    return result

def generate_dataset(num_images=1000):
    # Создаем директории, если они не существуют
    os.makedirs('data/with_flare', exist_ok=True)
    os.makedirs('data/without_flare', exist_ok=True)
    
    # Генерируем изображения
    for i in tqdm(range(num_images)):
        # Создаем базовое изображение
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Добавляем некоторые детали
        for _ in range(10):
            x1 = np.random.randint(0, 256)
            y1 = np.random.randint(0, 256)
            x2 = np.random.randint(0, 256)
            y2 = np.random.randint(0, 256)
            color = np.random.randint(0, 255, 3)
            cv2.line(image, (x1, y1), (x2, y2), color.tolist(), 2)
        
        # Сохраняем изображение без бликов
        cv2.imwrite(f'data/without_flare/image_{i:04d}.jpg', image)
        
        # Создаем версию с бликами
        image_with_flare = create_flare(image)
        cv2.imwrite(f'data/with_flare/image_{i:04d}.jpg', image_with_flare)

if __name__ == '__main__':
    generate_dataset() 