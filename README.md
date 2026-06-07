# SearchVector - локальний пошук документів з SVM

Веб-застосунок на Python, який імітує простий пошуковик для локальних `.txt` документів. Пошук працює на основі TF-IDF векторизації та методу опорних векторів (`LinearSVC`).

## Структура

```text
Term-Paper/
├── app.py
├── main.py
├── search_engine.py
├── requirements.txt
├── documents/
│   ├── svm_overview.txt
│   ├── search_engines.txt
│   ├── tfidf_explained.txt
│   ├── text_classification.txt
│   └── neural_networks.txt
├── templates/
│   └── index.html
└── static/
    ├── app.js
    └── style.css
```

## Запуск

Відкрийте термінал у папці:

```powershell
cd "C:\Users\Макс\PycharmProjects\Term-Paper"
```

Створіть та активуйте віртуальне середовище:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Встановіть залежності:

```powershell
pip install -r requirements.txt
```

Запустіть застосунок:

```powershell
python app.py
```

Після запуску відкрийте в браузері:

```text
http://127.0.0.1:5000
```

У PyCharm також можна запускати файл `main.py`.

## Як додати свої документи

1. Додайте `.txt` файли у папку `documents/`.
2. Перезапустіть `python app.py`.
3. Введіть пошуковий запит на головній сторінці.

## Як працює алгоритм

1. Застосунок читає всі `.txt` файли з папки `documents/`.
2. `TfidfVectorizer` перетворює документи у числові TF-IDF вектори.
3. `LinearSVC` навчається розрізняти документи як окремі класи.
4. Запит користувача теж перетворюється у TF-IDF вектор.
5. Для ранжування поєднуються оцінка SVM та косинусна подібність.
6. Результати показуються у веб-інтерфейсі з фрагментом тексту і відсотком релевантності.

## Приклади запитів

- `метод опорних векторів`
- `пошукові системи`
- `класифікація тексту`
- `TF-IDF`
- `нейронні мережі`
