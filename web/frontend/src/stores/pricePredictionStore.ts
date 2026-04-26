import { create } from "zustand";
import {
  PricePredictionRequestSchema,
  type PricePredictionRequest
} from "@/schemas/predictionSchema";

type PricePredictionStore = {
  params: PricePredictionRequest;
  setParam: <K extends keyof PricePredictionRequest>(
    key: K,
    value: PricePredictionRequest[K]
  ) => void;
  setParams: (params: Partial<PricePredictionRequest>) => void;
  reset: () => void;
};

const defaultParams = PricePredictionRequestSchema.parse({});

export const usePricePredictionStore = create<PricePredictionStore>((set) => ({
  params: defaultParams,
  setParam: (key, value) =>
    set((state) => ({
      params: { ...state.params, [key]: value }
    })),
  setParams: (params) =>
    set((state) => ({
      params: { ...state.params, ...params }
    })),
  reset: () => set({ params: defaultParams })
}));