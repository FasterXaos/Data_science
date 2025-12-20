import os
import pandas as pd
import requests
from urllib.parse import urlparse

# List of CSV files (must be in the same folder as the script)
csvFiles = [
    "observations-660819-Кряква.csv",
    "observations-660821-Домовый_воробей.csv",
    "observations-660822-Обыкновенный_скворец.csv",
    "observations-660823-Красный_кардинал.csv",
    "observations-660825-Черношапочная_гаичка.csv"
]

baseDir = "birds"
os.makedirs(baseDir, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

totalDownloaded = 0
totalSkipped = 0

for csvFile in csvFiles:
    if not os.path.exists(csvFile):
        print(f"Файл не найден: {csvFile} — пропуск")
        continue
    
    print(f"\nОбработка файла: {csvFile}")
    
    try:
        df = pd.read_csv(csvFile)
    except Exception as e:
        print(f"Ошибка чтения CSV {csvFile}: {e}")
        continue
    
    if 'image_url' not in df.columns or 'common_name' not in df.columns:
        print(f"В файле {csvFile} отсутствуют колонки 'image_url' или 'common_name' — пропуск")
        continue
    
    df = df.dropna(subset=['image_url'])
    df = df[df['image_url'].str.strip() != '']
    
    speciesName = df['common_name'].iloc[0].strip() if not df.empty else "Unknown"
    speciesDir = os.path.join(baseDir, speciesName)
    os.makedirs(speciesDir, exist_ok=True)
    
    print(f"Вид: {speciesName}")
    print(f"Количество записей с изображениями: {len(df)}")
    
    downloaded = 0
    skipped = 0
    
    for idx, row in df.iterrows():
        url = row['image_url'].strip()
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"  Ошибка {response.status_code} для {url}")
                skipped += 1
                continue
            
            contentType = response.headers.get('Content-Type', '')
            if 'jpeg' in contentType or url.endswith('.jpeg') or url.endswith('.jpg'):
                ext = '.jpg'
            elif 'png' in contentType:
                ext = '.png'
            else:
                parsed = urlparse(url)
                ext = os.path.splitext(parsed.path)[1]
                if ext not in ['.jpg', '.jpeg', '.png', '.gif']:
                    ext = '.jpg'  # fallback
            
            # File name: observation_id + ext
            obsId = row['id']
            filename = f"{obsId}{ext}"
            filepath = os.path.join(speciesDir, filename)
            
            if os.path.exists(filepath):
                skipped += 1
                continue
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"  Скачано {downloaded} изображений для {speciesName}...")
        
        except Exception as e:
            print(f"  Ошибка скачивания {url}: {e}")
            skipped += 1
    
    print(f"Готово {csvFile}: скачано {downloaded}, пропущено/ошибок {skipped}")
    
    totalDownloaded += downloaded
    totalSkipped += skipped

print("\n" + "="*50)
print(f"Всего скачано изображений: {totalDownloaded}")
print(f"Всего пропущено/ошибок: {totalSkipped}")
print(f"Изображения сохранены в папку '{baseDir}' по подпапкам с названиями видов.")