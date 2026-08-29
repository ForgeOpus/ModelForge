import "@fontsource-variable/inter";
import "@fontsource-variable/spline-sans-mono";
import "./styles/theme.css";
import "./styles/base.css";
import "./styles/utilities.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { AppRouter } from "./router";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AppRouter />
  </StrictMode>,
);
