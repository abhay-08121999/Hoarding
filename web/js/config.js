/*
 * config.js — the one place to switch on real posters and trailers.
 *
 * Get a free key at https://www.themoviedb.org/settings/api and paste it
 * below. Either kind works:
 *   - "API Key"               (32 hex characters)
 *   - "API Read Access Token" (long string starting with "eyJ")
 *
 * Two ways to use it:
 *   1. Paste it here  -> works everywhere: double-clicking index.html,
 *      python app.py, or any static host. The key is visible in the page
 *      source, which TMDB permits for v3 keys but keep it in mind for a
 *      public deployment.
 *   2. Leave this blank and run  TMDB_API_KEY=... python app.py  (or put it
 *      in a .env file). The Flask server then calls TMDB for you and the key
 *      never reaches the browser.
 *
 * If both are set, the Flask server is used first.
 */
window.APP_CONFIG = {
  TMDB_API_KEY: "e409fc67d2075fbbac09da98d072ad80",

  // Only change these to point at a proxy or a test double.
  TMDB_API_BASE: "https://api.themoviedb.org/3",
  TMDB_IMAGE_BASE: "https://image.tmdb.org/t/p/",

  // Set false to skip the /api/metadata call on file:// or static hosting.
  USE_SERVER_PROXY: true,
};