import PriceChart from "@/components/charts/PriceChart";
import { type PricePredictionResponse } from "@/schemas/predictionSchema";

type PredictionChartProps = {
  data: PricePredictionResponse;
};

const PredictionChart = ({ data }: PredictionChartProps) => {
  return (
    <div className='h-[500px]'>
      <PriceChart data={data} />
    </div>
  );
};

export default PredictionChart;
