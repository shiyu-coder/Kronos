import { create } from "zustand";

type ChartUIStore = {
  showVolume: boolean;
  setShowVolume: (show: boolean) => void;
};

export const useChartUIStore = create<ChartUIStore>((set) => ({
  showVolume: false,
  setShowVolume: (show) => set({ showVolume: show })
}));