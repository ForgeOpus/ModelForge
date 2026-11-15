import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { captureStartupToken } from "@/lib/api";
import { App } from "@/App";
import "./index.css";

captureStartupToken();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
