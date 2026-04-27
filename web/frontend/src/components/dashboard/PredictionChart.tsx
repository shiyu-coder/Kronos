import PriceChart from "@/components/charts/PriceChart";
import { type PricePredictionResponse } from "@/schemas/predictionSchema";

type PredictionChartProps = {
  data: PricePredictionResponse;
};

const PredictionChart = ({ data }: PredictionChartProps) => {
  return <PriceChart data={data} />;
};

export default PredictionChart;
