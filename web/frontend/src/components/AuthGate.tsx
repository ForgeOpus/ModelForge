import { useState } from "react";
import { KeyRound } from "lucide-react";
import { setToken } from "@/lib/api";
import { useStore } from "@/lib/store";
import { Button } from "./ui/button";
import { Input, Label } from "./ui/field";
import { Lockup } from "./Logo";

/* What the screen says when the server will not talk to us.

   ModelForge prints a one-time token at startup and the browser keeps it in
   local storage. Open the app in a different browser -- or clear the storage --
   and every request comes back 401. The old behaviour was to leave empty
   skeletons on screen indefinitely, which reads as a broken app rather than a
   missing key. */
export function AuthGate() {
  const { reload } = useStore();
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!value.trim()) return;
    setBusy(true);
    setToken(value.trim());
    await reload();
    setBusy(false);
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-ground px-6">
      <div className="w-full max-w-[380px]">
        <Lockup duo size="md" className="mb-7" />
        <h1 className="font-display text-[19px] font-semibold">This session needs a token</h1>
        <p className="mt-2 text-[13px] leading-relaxed text-dim">
          ModelForge prints a link when it starts, with a one-time token in it. Open that link, or
          paste the token here.
        </p>
        <pre className="mt-4 overflow-x-auto rounded-sm border border-line bg-surface p-3 font-mono text-[11.5px] text-dim">
          modelforge run
        </pre>
        <form onSubmit={submit} className="mt-4">
          <Label htmlFor="token">Startup token</Label>
          <div className="flex gap-2">
            <Input
              id="token"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="paste the token"
              autoComplete="off"
            />
            <Button type="submit" variant="primary" disabled={busy || !value.trim()}>
              <KeyRound /> Connect
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
