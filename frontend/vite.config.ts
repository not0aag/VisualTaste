import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Loopback only, like the backend. Nothing in this project binds to 0.0.0.0.
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      // The API is proxied in development so the browser sees one origin and
      // image URLs like /api/images/<id>/display work unchanged in both the
      // dev server and a production build.
      '/api': {
        // Configurable so the end-to-end run can point at its own sandboxed
        // backend on a different port instead of a live dev server.
        target: process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000',
        changeOrigin: false,
      },
    },
  },
  build: { outDir: 'dist', sourcemap: true },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.ts'],
    include: ['tests/**/*.test.{ts,tsx}'],
  },
})
