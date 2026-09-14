(() => {
  const key = "transformer-theme";
  const valid = value => value === "dark" ? "dark" : "light";
  let theme = "light";
  try {
    theme = valid(localStorage.getItem(key));
  } catch {
    // Storage may be unavailable in private or restricted browser sessions.
  }
  document.documentElement.dataset.theme = theme;
  document.addEventListener("DOMContentLoaded", () => {
    const select = document.getElementById("theme-select");
    select.value = theme;
    select.addEventListener("change", () => {
      theme = valid(select.value);
      document.documentElement.dataset.theme = theme;
      try {
        localStorage.setItem(key, theme);
      } catch {
        // Theme switching remains functional without persistence.
      }
    });
  });
})();
