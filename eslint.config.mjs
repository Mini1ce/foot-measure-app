import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Self-hosted OpenCV vendor bundle (generated/minified, about 8 MB).
    "public/opencv.js",
    // Self-hosted MediaPipe Emscripten loaders; generated vendor code.
    "public/mediapipe/wasm/*.js",
  ]),
]);

export default eslintConfig;
