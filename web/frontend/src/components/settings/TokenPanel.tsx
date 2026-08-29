import { useState } from "react";
import { useToken } from "../../hooks";
import { Field, Input, Button } from "../ui";

function mask(token: string): string {
  if (!token) return "not set";
  if (token.length <= 8) return "••••";
  return `${token.slice(0, 4)}••••${token.slice(-4)}`;
}

export function TokenPanel() {
  const { token, save, clear } = useToken();
  const [draft, setDraft] = useState("");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
      <p className="mono" style={{ color: "var(--text-2)" }}>{mask(token)}</p>

      <Field label="API token" htmlFor="token-input">
        <Input
          id="token-input"
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Paste token"
        />
      </Field>

      <div className="cluster">
        <Button
          variant="primary"
          onClick={() => {
            save(draft);
            setDraft("");
          }}
        >
          Replace token
        </Button>
        <Button variant="ghost" onClick={clear}>
          Clear token
        </Button>
      </div>

      <p style={{ fontSize: "var(--step--1)", color: "var(--text-3)" }}>
        The token is printed by <span className="mono">modelforge run</span> at startup as a{" "}
        <span className="mono">?token=…</span> URL and is required for every API call.
      </p>
    </div>
  );
}
