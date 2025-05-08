// src/main.jsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import { Provider } from "react-redux";
import { persistor, store } from "./redux/store.js";
import { PersistGate } from "redux-persist/integration/react";

console.log("Main.jsx: Starting application");

// Define global for compatibility
window.global = window;

const rootElement = document.getElementById("root");
console.log("Main.jsx: Root element found?", !!rootElement);

if (!rootElement) {
  console.error("Main.jsx: Root element not found");
} else {
  const root = createRoot(rootElement);
  console.log("Main.jsx: Creating root");

  root.render(
    <StrictMode>
      <Provider store={store}>
        <PersistGate loading={null} persistor={persistor}>
          <App />
        </PersistGate>
      </Provider>
    </StrictMode>
  );
  console.log("Main.jsx: App rendered");
}
