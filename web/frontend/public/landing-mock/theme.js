(function () {
  var KEY = "mf-theme";
  function stored() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function system() { return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"; }
  var fromUrl = new URLSearchParams(location.search).get("theme");
  document.documentElement.dataset.theme = fromUrl || stored() || system();

  window.mfSetTheme = function (t) {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem(KEY, t); } catch (e) {}
    window.dispatchEvent(new CustomEvent("mf-theme", { detail: t }));
  };

  document.addEventListener("click", function (e) {
    if (!e.target.closest("[data-theme-toggle]")) return;
    window.mfSetTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
})();
