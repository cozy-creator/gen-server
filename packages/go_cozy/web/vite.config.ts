import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    emptyOutDir: true,
    minify: true,
    rollupOptions: {
      // Disable code splitting
      output: {
        manualChunks: undefined,
        // Name of bundle file
        entryFileNames: 'bundle.js',
      },
    },
  },
});
