import { useCallback, useState } from "react";
import { hasToken, resolveToken, setToken } from "../api/client";

export function useToken(): {
  token: string;
  hasToken: boolean;
  save: (t: string) => void;
  clear: () => void;
} {
  const [token, setTokenState] = useState<string>(() => resolveToken());

  const save = useCallback((t: string) => {
    setToken(t);
    setTokenState(resolveToken());
  }, []);

  const clear = useCallback(() => {
    setToken("");
    setTokenState(resolveToken());
  }, []);

  return { token, hasToken: hasToken(), save, clear };
}
