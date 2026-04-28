import { forwardRef } from "react";
import PriceChart from "@/components/charts/PriceChart";
import { type PricePredictionResponse } from "@/schemas/predictionSchema";

type PredictionChartProps = {
  data: PricePredictionResponse;
};

const PredictionChart = forwardRef<HTMLDivElement, PredictionChartProps>(
  ({ data }, ref) => {
    return (
      <div ref={ref} className='h-[500px]'>
        <PriceChart data={data} />
      </div>
    );
  }
);

PredictionChart.displayName = "PredictionChart";

export default PredictionChart;
