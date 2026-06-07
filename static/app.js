const homePage = document.getElementById("home-page");
const resultsPage = document.getElementById("results-page");
const mainInput = document.getElementById("main-input");
const resultsInput = document.getElementById("results-input");
const resultsList = document.getElementById("results-list");
const resultsMeta = document.getElementById("results-meta");
const noResults = document.getElementById("no-results");
const loader = document.getElementById("loader");

function showPage(page) {
  document.querySelectorAll(".page").forEach((item) => item.classList.remove("active"));
  page.classList.add("active");
}

function setLoading(value) {
  loader.classList.toggle("hidden", !value);
}

async function doSearch(query) {
  const cleanQuery = query.trim();
  if (!cleanQuery) {
    return;
  }

  setLoading(true);
  showPage(resultsPage);
  resultsInput.value = cleanQuery;
  resultsList.innerHTML = "";
  resultsMeta.textContent = "";
  noResults.classList.add("hidden");

  try {
    const response = await fetch(`/search?q=${encodeURIComponent(cleanQuery)}`);
    const data = await response.json();
    renderResults(data);
  } catch (error) {
    resultsMeta.textContent = "Помилка з'єднання із сервером.";
    console.error(error);
  } finally {
    setLoading(false);
  }
}

function renderResults(data) {
  if (data.total === 0) {
    resultsMeta.textContent = `За запитом "${data.query}" нічого не знайдено.`;
    noResults.classList.remove("hidden");
    return;
  }

  resultsMeta.innerHTML = `Знайдено <strong>${data.total}</strong> результатів за запитом <strong>"${escapeHtml(data.query)}"</strong>`;

  data.results.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "result-card";
    card.innerHTML = `
      <div class="rank">${index + 1}</div>
      <h2>${escapeHtml(item.title)}</h2>
      <p class="file">${escapeHtml(item.filename)}</p>
      <p class="snippet">${item.snippet}</p>
      <div class="footer">
        <span>Релевантність: ${item.score}%</span>
        <span>${item.word_count} слів</span>
      </div>
      <div class="bar"><div style="width: ${Math.min(item.score, 100)}%"></div></div>
    `;
    resultsList.appendChild(card);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

document.getElementById("main-search-btn").addEventListener("click", () => doSearch(mainInput.value));
document.getElementById("results-search-btn").addEventListener("click", () => doSearch(resultsInput.value));
document.getElementById("home-btn").addEventListener("click", () => {
  showPage(homePage);
  mainInput.focus();
});

mainInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    doSearch(mainInput.value);
  }
});

resultsInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    doSearch(resultsInput.value);
  }
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    mainInput.value = chip.dataset.query;
    doSearch(chip.dataset.query);
  });
});

mainInput.focus();
