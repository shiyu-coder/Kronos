import { create } from "zustand";
import {
  PricePredictionRequestSchema,
  type PricePredictionRequest,
  type PricePredictionResponse,
  type PricePredictionBatchResponse
} from "@/schemas/predictionSchema";

export type PredictionError = {
  title: string;
  message: string;
  code?: string;
  timestamp: Date;
};

export type PredictionMode = "single" | "batch";

const MAX_BATCH_SIZE = 20;

type PricePredictionStore = {
  // --- Mode ---
  mode: PredictionMode;
  setMode: (mode: PredictionMode) => void;

  // --- Single prediction params ---
  params: PricePredictionRequest;
  setParam: <K extends keyof PricePredictionRequest>(key: K, value: PricePredictionRequest[K]) => void;
  setParams: (params: Partial<PricePredictionRequest>) => void;

  // --- Batch shared params (independent from single) ---
  batchParams: PricePredictionRequest;
  setBatchParam: <K extends keyof PricePredictionRequest>(key: K, value: PricePredictionRequest[K]) => void;
  setBatchParams: (params: Partial<PricePredictionRequest>) => void;

  // --- Single result ---
  result: PricePredictionResponse | null;
  setResult: (result: PricePredictionResponse | null) => void;

  // ----------------------------------------

  // --- Batch items (per-request overrides) ---
  batchItems: Partial<PricePredictionRequest>[];
  addBatchItem: () => void;
  updateBatchItem: (index: number, partial: Partial<PricePredictionRequest>) => void;
  removeBatchItem: (index: number) => void;

  // --- Batch results ---
  batchResults: PricePredictionBatchResponse | null;
  setBatchResults: (results: PricePredictionBatchResponse | null) => void;
  activeResultIndex: number;
  setActiveResultIndex: (index: number) => void;

  // --- Loading ---
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;

  // --- Error ---
  error: PredictionError | null;
  setError: (error: PredictionError) => void;
  clearError: () => void;

  // --- Convenience: get the active params for the current mode ---
  getActiveParams: () => PricePredictionRequest;

  // --- Reset ---
  reset: () => void;
  resetResult: () => void;
};

const defaultParams = PricePredictionRequestSchema.parse({});
const defaultBatchItem = defaultParams;

export const usePricePredictionStore = create<PricePredictionStore>((set, get) => ({
  // Mode
  mode: "single",
  setMode: (mode) => set({ mode }),

  // Single params
  params: { ...defaultParams },
  setParam: (key, value) =>
    set((state) => ({
      params: { ...state.params, [key]: value }
    })),
  setParams: (params) =>
    set((state) => ({
      params: { ...state.params, ...params }
    })),

  // ----------------------------------------

  // Batch shared params (independent copy)
  batchParams: { ...defaultParams },
  setBatchParam: (key, value) =>
    set((state) => ({
      batchParams: { ...state.batchParams, [key]: value }
    })),
  setBatchParams: (params) =>
    set((state) => ({
      batchParams: { ...state.batchParams, ...params }
    })),

  // Single result
  result: null,
  setResult: (result) => set({ result }),

  // Batch items
  batchItems: [{ ...defaultBatchItem }],
  addBatchItem: () =>
    set((state) => {
      if (state.batchItems.length >= MAX_BATCH_SIZE) return state;
      return { batchItems: [...state.batchItems, { ...defaultBatchItem }] };
    }),
  updateBatchItem: (index, partial) =>
    set((state) => {
      const updated = [...state.batchItems];
      updated[index] = { ...updated[index], ...partial };
      return { batchItems: updated };
    }),
  removeBatchItem: (index) =>
    set((state) => {
      if (state.batchItems.length <= 1) return state;
      const updated = state.batchItems.filter((_, i) => i !== index);
      const newActiveIndex = Math.min(state.activeResultIndex, updated.length - 1);
      return { batchItems: updated, activeResultIndex: newActiveIndex };
    }),

  // Batch results
  batchResults: null,
  setBatchResults: (results) => set({ batchResults: results }),
  activeResultIndex: 0,
  setActiveResultIndex: (index) => set({ activeResultIndex: index }),

  // Loading
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),

  // Error
  error: null,
  setError: (error) => set({ error }),
  clearError: () => set({ error: null }),

  // Convenience: returns the correct params for the current mode
  getActiveParams: () => {
    const state = get();
    return state.mode === "batch" ? state.batchParams : state.params;
  },

  // Reset
  reset: () =>
    set({
      params: { ...defaultParams },
      batchParams: { ...defaultParams },
      result: null,
      batchItems: [{ ...defaultBatchItem }],
      batchResults: null,
      activeResultIndex: 0,
      error: null
    }),
  resetResult: () => set({ result: null, batchResults: null, activeResultIndex: 0, error: null })
}));
