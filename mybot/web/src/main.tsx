import React from "react";
import ReactDOM from "react-dom/client";
import { App as AntdApp, ConfigProvider, message, theme } from "antd";
import Dashboard from "./App";
import "./index.css";

/** Toasts sit below the sticky header (~64px + margin). */
message.config({ top: 84, maxCount: 4, duration: 3 });

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          colorPrimary: "#0d9488",
          borderRadiusLG: 12,
          fontFamily:
            "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
        },
      }}
    >
      <AntdApp message={{ top: 84, maxCount: 4 }} notification={{ placement: "bottomRight" }}>
        <Dashboard />
      </AntdApp>
    </ConfigProvider>
  </React.StrictMode>
);
