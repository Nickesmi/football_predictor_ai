import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  // When deploying to GitHub Pages the site lives under /<repo-name>/.
  // VITE_BASE_PATH is set in the CI workflow; defaults to '/' for local dev.
  base: process.env.VITE_BASE_PATH ?? '/',
})
