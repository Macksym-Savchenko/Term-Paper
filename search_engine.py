import html
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


class SearchEngine:
    """
    Локальний пошуковий рушій на основі TF-IDF та методу опорних векторів.

    Документи з папки documents/ перетворюються на TF-IDF-вектори. SVM
    навчається розрізняти документи як окремі класи, а під час пошуку
    запит ранжується за оцінкою SVM і косинусною подібністю.
    """

    MIN_PREFIX_LEN = 3
    MIN_ROOT_LEN = 5

    def __init__(self, documents_folder: str):
        self.documents_folder = documents_folder
        self.documents = {}
        self.filenames = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.svm = None
        self.is_indexed = False

    def index_documents(self) -> None:
        """Load .txt files, build a TF-IDF matrix, and train the SVM model."""
        self._load_documents()

        if not self.documents:
            print("[!] У папці documents/ немає текстових файлів")
            return

        self.filenames = list(self.documents.keys())
        texts = [self.documents[name] for name in self.filenames]

        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self._train_svm()
        self.is_indexed = True

        print(f"[OK] Проіндексовано документів: {len(self.documents)}")
        if self.svm is not None:
            print("[OK] SVM-модель навчена")

    def search(self, query: str, top_n: int = 10) -> list[dict]:
        """Return the most relevant documents for a query."""
        if not self.is_indexed or not query.strip():
            return []

        expanded_query = self._expand_query(query)
        query_vector = self.vectorizer.transform([expanded_query])
        if query_vector.nnz == 0:
            return self._substring_search(query, top_n)

        cosine_scores = self._cosine_similarity(query_vector, self.tfidf_matrix)

        if self.svm is not None:
            svm_scores = np.asarray(self.svm.decision_function(query_vector)).ravel()
            if len(svm_scores) == len(self.filenames):
                scores = (
                    0.7 * self._minmax_normalize(svm_scores)
                    + 0.3 * self._minmax_normalize(cosine_scores)
                )
            else:
                scores = self._minmax_normalize(cosine_scores)
        else:
            scores = self._minmax_normalize(cosine_scores)

        results = self._collect_results(scores, query, top_n)
        if results:
            return results

        return self._substring_search(query, top_n)

    def _expand_query(self, query: str) -> str:
        """Expand the query with similar vocabulary words by shared word root."""
        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            return query

        expanded: list[str] = []
        seen: set[str] = set()

        def add_term(term: str) -> None:
            if term not in seen:
                seen.add(term)
                expanded.append(term)

        for token in tokens:
            add_term(token)

        vocabulary_words = self._vocabulary_words()
        for token in tokens:
            if len(token) < self.MIN_PREFIX_LEN:
                continue

            for word in vocabulary_words:
                if self._share_word_root(token, word):
                    add_term(word)

        return " ".join(expanded)

    def _vocabulary_words(self) -> set[str]:
        words: set[str] = set()
        for term in self.vectorizer.get_feature_names_out():
            for part in term.lower().split():
                words.add(part)
        return words

    @classmethod
    def _common_prefix_length(cls, left: str, right: str) -> int:
        limit = min(len(left), len(right))
        index = 0
        while index < limit and left[index] == right[index]:
            index += 1
        return index

    @classmethod
    def _share_word_root(cls, left: str, right: str) -> bool:
        if len(left) < cls.MIN_PREFIX_LEN or len(right) < cls.MIN_PREFIX_LEN:
            return False

        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        if longer.startswith(shorter) and len(shorter) >= cls.MIN_ROOT_LEN:
            return True

        return cls._common_prefix_length(left, right) >= cls.MIN_ROOT_LEN

    def _collect_results(
        self, scores: np.ndarray, query: str, top_n: int
    ) -> list[dict]:
        ranked_indices = np.argsort(scores)[::-1]
        results = []

        for index in ranked_indices[:top_n]:
            score = float(scores[index])
            if score < 0.05:
                break

            filename = self.filenames[index]
            text = self.documents[filename]
            results.append(
                {
                    "filename": filename,
                    "title": self._extract_title(filename, text),
                    "snippet": self._extract_snippet(text, query),
                    "score": round(score * 100, 1),
                    "word_count": len(text.split()),
                }
            )

        return results

    def _substring_search(self, query: str, top_n: int) -> list[dict]:
        """Fallback search when TF-IDF cannot map the query to known terms."""
        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            return []

        scored: list[tuple[float, str]] = []

        for filename in self.filenames:
            text = self.documents[filename]
            text_lower = text.lower()
            score = 0.0

            for token in tokens:
                if token in text_lower:
                    score += 2.0

                for word in re.findall(r"\w+", text_lower):
                    if self._share_word_root(token, word):
                        score += 1.0

            if score > 0:
                scored.append((score, filename))

        scored.sort(key=lambda item: item[0], reverse=True)
        max_score = scored[0][0] if scored else 1.0
        results = []

        for score, filename in scored[:top_n]:
            text = self.documents[filename]
            results.append(
                {
                    "filename": filename,
                    "title": self._extract_title(filename, text),
                    "snippet": self._extract_snippet(text, query),
                    "score": round((score / max_score) * 100, 1),
                    "word_count": len(text.split()),
                }
            )

        return results

    def get_all_documents(self) -> list[dict]:
        """Return metadata for all indexed documents."""
        return [
            {
                "filename": filename,
                "title": self._extract_title(filename, text),
                "word_count": len(text.split()),
                "preview": text[:120] + "..." if len(text) > 120 else text,
            }
            for filename, text in self.documents.items()
        ]

    def _load_documents(self) -> None:
        self.documents = {}
        self.filenames = []
        self.is_indexed = False

        if not os.path.exists(self.documents_folder):
            os.makedirs(self.documents_folder)
            return

        for filename in sorted(os.listdir(self.documents_folder)):
            if not filename.lower().endswith(".txt"):
                continue
            if re.match(r"^\d{4}_", filename):
                continue

            path = os.path.join(self.documents_folder, filename)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    text = file.read().strip()
            except OSError as error:
                print(f"[!] Не вдалося прочитати {filename}: {error}")
                continue

            if text:
                self.documents[filename] = text

    def _train_svm(self) -> None:
        if len(self.filenames) < 2:
            self.svm = None
            return

        labels = np.arange(len(self.filenames))
        self.svm = LinearSVC(C=1.0, max_iter=5000)
        self.svm.fit(self.tfidf_matrix, labels)

    @staticmethod
    def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
        minimum = float(scores.min())
        maximum = float(scores.max())
        if maximum - minimum < 1e-9:
            return np.ones_like(scores, dtype=float) * 0.5
        return (scores - minimum) / (maximum - minimum)

    @staticmethod
    def _cosine_similarity(query_vector, document_matrix) -> np.ndarray:
        dot_product = (document_matrix @ query_vector.T).toarray().ravel()
        query_norm = np.sqrt((query_vector @ query_vector.T).toarray()[0][0])
        document_norms = np.sqrt(np.asarray(document_matrix.power(2).sum(axis=1)).ravel())

        denominator = document_norms * query_norm
        denominator[denominator == 0] = 1e-10
        return dot_product / denominator

    @staticmethod
    def _extract_title(filename: str, text: str) -> str:
        first_line = text.splitlines()[0].strip()
        if 5 < len(first_line) < 120:
            return first_line
        return filename.replace(".txt", "").replace("_", " ").title()

    @classmethod
    def _find_match_position(cls, text_lower: str, token: str) -> int:
        position = text_lower.find(token)
        if position != -1:
            return position

        for match in re.finditer(r"\w+", text_lower):
            if cls._share_word_root(match.group(0), token):
                return match.start()

        return -1

    @classmethod
    def _extract_snippet(cls, text: str, query: str, length: int = 220) -> str:
        query_words = re.findall(r"\w+", query.lower())
        text_lower = text.lower()

        start = 0
        for word in query_words:
            position = cls._find_match_position(text_lower, word)
            if position != -1:
                start = max(0, position - 70)
                break

        snippet = text[start : start + length].strip()
        if start > 0 and " " in snippet:
            snippet = "..." + snippet[snippet.index(" ") :]
        if len(text) > start + length:
            snippet += "..."

        safe_snippet = html.escape(snippet)
        for word in query_words:
            if len(word) >= cls.MIN_ROOT_LEN:
                pattern = re.compile(
                    re.escape(word[: cls.MIN_ROOT_LEN]) + r"\w*",
                    re.IGNORECASE,
                )
            elif len(word) >= cls.MIN_PREFIX_LEN:
                pattern = re.compile(
                    r"\w*" + re.escape(word) + r"\w*",
                    re.IGNORECASE,
                )
            else:
                pattern = re.compile(re.escape(word), re.IGNORECASE)

            safe_snippet = pattern.sub(
                lambda match: f"<mark>{match.group(0)}</mark>",
                safe_snippet,
            )

        return safe_snippet
