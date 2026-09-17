const form = document.getElementById("profileForm");
const statusEl = document.getElementById("status");
const resultsTable = document.getElementById("results");
const resultsBody = resultsTable.querySelector("tbody");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const apiBase = document.getElementById("apiBase").value.replace(/\/$/, "");
  const payload = {
    Age: Number(document.getElementById("age").value),
    State: document.getElementById("state").value,
    Income: Number(document.getElementById("income").value),
    Purchases: Number(document.getElementById("purchases").value),
    LastPurchaseDate: document.getElementById("lastPurchaseDate").value || null,
    Review: document.getElementById("review").value,
  };

  statusEl.textContent = "Calling the API...";
  statusEl.classList.remove("error");
  resultsTable.hidden = true;

  try {
    const response = await fetch(`${apiBase}/similar-customers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`API returned ${response.status}`);
    }

    const matches = await response.json();
    renderResults(matches);
    statusEl.textContent = `Found ${matches.length} similar customer(s).`;
  } catch (err) {
    statusEl.textContent = `Request failed: ${err.message}`;
    statusEl.classList.add("error");
  }
});

function renderResults(matches) {
  resultsBody.innerHTML = "";
  for (const match of matches) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${match.CustomerID}</td>
      <td>${match.State}</td>
      <td>${match.Age}</td>
      <td>${match.Income}</td>
      <td>${match.Purchases}</td>
      <td>${match.Review ?? ""}</td>
      <td>${match.distance.toFixed(4)}</td>
    `;
    resultsBody.appendChild(row);
  }
  resultsTable.hidden = matches.length === 0;
}
