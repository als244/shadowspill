"use strict";
const table = document.querySelector("#configurations tbody");
if (table) {
  const rows = [...table.rows];
  const controls = Object.fromEntries(["search", "model", "budget", "scope", "sort"].map(id => [id, document.getElementById(id)]));
  // A matrix-only or archived subfolder should show its own entries immediately.
  if (rows.length && rows.every(r => ["Archive", "Earlier cut"].includes(r.dataset.category))) controls.scope.value = "all";
  function filter() {
    const query = controls.search.value.trim().toLowerCase();
    const scope = controls.scope.value;
    const ordering = controls.sort.value;
    const sorted = [...rows];
    if (ordering !== "library") sorted.sort((a, b) => {
      const x = Number(a.dataset.rate), y = Number(b.dataset.rate);
      if (x < 0 || y < 0) return x < 0 ? (y < 0 ? 0 : 1) : -1;
      return ordering === "fastest" ? y - x : x - y;
    });
    let matches = 0;
    for (const row of sorted) {
      const d = row.dataset;
      const earlier = ["Archive", "Earlier cut"].includes(d.category);
      const selected = scope === "all" || (scope === "active" && !earlier) || (scope === "earlier" && earlier) || scope === d.category;
      row.hidden = !(selected && (!query || d.search.includes(query)) && (!controls.model.value || d.model === controls.model.value) && (!controls.budget.value || d.budget === controls.budget.value));
      if (!row.hidden) matches++;
      table.append(row);
    }
    document.getElementById("match-count").textContent = `${matches} of ${rows.length} configurations shown`;
    document.getElementById("empty").hidden = matches !== 0;
  }
  Object.values(controls).forEach(control => control.addEventListener("input", filter));
  filter();
}
document.getElementById("refresh")?.addEventListener("click", () => location.reload());
const player = document.getElementById("player");
if (player) {
  const choices = [...document.querySelectorAll(".choose-video")];
  choices[0]?.classList.add("selected");
  choices.forEach(button => button.addEventListener("click", () => {
    player.pause();
    player.src = button.dataset.src;
    player.load();
    document.getElementById("now-playing").textContent = button.dataset.label;
    choices.forEach(item => item.classList.toggle("selected", item === button));
  }));
}
