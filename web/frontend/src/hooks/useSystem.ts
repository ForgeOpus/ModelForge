import { api } from "../api/client";
import { useApi } from "./useApi";

export function useSystem() {
  return useApi(() => api.system(), []);
}
