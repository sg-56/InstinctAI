import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react({
      // Add React plugin options for debugging
      jsxRuntime: "automatic",
      babel: {
        plugins: [
          ["@babel/plugin-transform-react-jsx", { runtime: "automatic" }],
        ],
      },
    }),
  ],
  server: {
    proxy: {},
    hmr: {
      overlay: true,
    },
  },
  resolve: {
    alias: {
      "@": "/src", // Ensure this alias is correctly pointing to 'src'
    },
  },
  define: {
    global: {},
  },
  build: {
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom"],
        },
      },
    },
  },
  logLevel: "info",
});
