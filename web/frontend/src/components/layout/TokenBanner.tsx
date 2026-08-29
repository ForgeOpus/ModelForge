import { useState } from "react";
import { useToken } from "../../hooks";
import { Button, Callout, Input } from "../ui";

export function TokenBanner() {
  const { hasToken, save } = useToken();
  const [value, setValue] = useState("");

  if (hasToken) return null;

  return (
    <Callout tone="warning" title="No access token">
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
        <p>
          When you start the server, <code className="mono">modelforge run</code> prints a URL with
          a <code className="mono">?token=&hellip;</code> value on the end. Copy that token and paste
          it here to connect.
        </p>
        <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
          <Input
            aria-label="Access token"
            placeholder="Paste token"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            style={{ flex: "1 1 240px" }}
          />
          <Button
            variant="primary"
            size="sm"
            disabled={value.trim() === ""}
            onClick={() => save(value.trim())}
          >
            Save
          </Button>
        </div>
      </div>
    </Callout>
  );
}
