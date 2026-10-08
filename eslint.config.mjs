// Repo-wide ESLint config (flat config, ESLint 10).
// The viewer and mobile packages may extend or override this in later steps.
import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import tseslint from "typescript-eslint";

export default defineConfig([
  globalIgnores([
    "**/node_modules/",
    "**/dist/",
    "**/build/",
    "**/coverage/",
    "mobile/.expo/",
    "mobile/android/",
    "mobile/ios/",
    "data/",
    ".venv/",
  ]),
  js.configs.recommended,
  tseslint.configs.recommended,
]);
