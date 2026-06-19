"""PyCharm entry point for the SearchVector web application."""

from app import app, engine


if __name__ == "__main__":
    print("=" * 60)
    print(" SearchVector запущено")
    print(f" Проіндексовано документів: {len(engine.documents)}")
    print(" Відкрийте у браузері: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=False)
