import { forwardRef } from "react";
import PriceChart from "@/components/charts/PriceChart";
import { Switch } from "@/components/ui/switch";
import { useChartUIStore } from "@/stores/chartUIStore";
import { type PricePredictionResponse } from "@/schemas/predictionSchema";

type PredictionChartProps = {
  data: PricePredictionResponse;
};

const PredictionChart = forwardRef<HTMLDivElement, PredictionChartProps>(({ data }, ref) => {
  const showVolume = useChartUIStore((state) => state.showVolume);
  const setShowVolume = useChartUIStore((state) => state.setShowVolume);

  return (
    <div ref={ref} className='h-[520px]'>
      <div className='flex justify-end gap-2 pb-2'>
        <Switch checked={showVolume} onCheckedChange={setShowVolume} className='cursor-pointer' />
        <span className='text-sm text-muted-foreground'>Volume</span>
      </div>
      <PriceChart data={data} showVolume={showVolume} />
    </div>
  );
});

PredictionChart.displayName = "PredictionChart";

export default PredictionChart;
