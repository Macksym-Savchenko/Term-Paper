import os

from flask import Flask, jsonify, render_template, request

from search_engine import SearchEngine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")

app = Flask(__name__)
engine = SearchEngine(DOCUMENTS_DIR)
engine.index_documents()


@app.route("/")
def index():
    return render_template("index.html", doc_count=len(engine.documents))


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"query": "", "total": 0, "results": []})

    results = engine.search(query, top_n=10)
    return jsonify({"query": query, "total": len(results), "results": results})


@app.route("/documents")
def documents():
    return jsonify({"documents": engine.get_all_documents()})


if __name__ == "__main__":
    print("=" * 60)
    print(" SearchVector запущено")
    print(f" Проіндексовано документів: {len(engine.documents)}")
    print(" Відкрийте у браузері: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True)
