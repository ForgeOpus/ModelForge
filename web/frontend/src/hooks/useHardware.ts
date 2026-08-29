import { api } from "../api/client";
import { useApi } from "./useApi";

export function useHardware() {
  return useApi(() => api.hardware(), []);
}
