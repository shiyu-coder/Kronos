"use client";

import {
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  ReferenceLine,
  Bar
} from "recharts";
import { type PricePredictionResponse } from "@/schemas/predictionSchema";

type PriceChartProps = {
  data: PricePredictionResponse | null;
  showVolume?: boolean;
};

const PriceChart = ({ data, showVolume = true }: PriceChartProps) => {
  if (!data) {
    return null;
  }

  type ChartPoint = {
    timestamp: string;
    price: number | null;
    prediction: number | null;
    volume: number;
    type: string;
  };

  const historyData: ChartPoint[] = data.history.map((item) => ({
    timestamp: item.timestamps,
    price: item.close,
    prediction: null,
    volume: item.volume,
    type: "history"
  }));

  const predictionData: ChartPoint[] = data.prediction.map((item) => ({
    timestamp: item.timestamps,
    price: null,
    prediction: item.close,
    volume: item.volume,
    type: "prediction"
  }));

  const firstPrediction = data.prediction[0];

  // Override the first prediction item to include price value
  predictionData[0].price = firstPrediction.close;

  const allData = [...historyData, ...predictionData];

  const allPrices = allData.map((d) => d.price ?? d.prediction).filter((p) => p !== null) as number[];

  const minPrice = Math.min(...allPrices);
  const maxPrice = Math.max(...allPrices);

  const padding = (maxPrice - minPrice) * 0.1; // 10% padding

  const yMin = Math.floor(minPrice - padding);
  const yMax = Math.ceil(maxPrice + padding);

  const allVolumes = allData.map((d) => d.volume).filter((v) => v !== null && v !== undefined) as number[];
  const maxVolume = Math.max(...allVolumes);
  const volumeDomain = [0, maxVolume * 1.08]; // 8% padding

  const predictionStartDate = data.prediction_start_timestamp;

  return (
    <div className='size-full border'>
      <ResponsiveContainer width='100%' height='100%'>
        <ComposedChart data={allData} margin={{ top: 30, right: 30, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray='3 3' vertical={false} stroke='rgba(255,255,255,0.1)' />

          <XAxis
            dataKey='timestamp'
            tickLine={false}
            axisLine={false}
            tickMargin={20}
            tick={{ fontSize: 12 }}
            interval={Math.floor(allData.length / 7)} // Show ~6 labels across chart
            tickFormatter={(value) => {
              const date = new Date(value);
              return date.toISOString().split("T")[0]; // YYYY-MM-DD format
            }}
            padding={{ left: 40, right: 0 }}
          />

          {/* Price Y-axis */}
          <YAxis
            domain={[yMin, yMax]}
            tickLine={false}
            axisLine={false}
            tickMargin={10}
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => `$${value.toLocaleString()}`}
          />

          {/* Volume Y-axis */}
          {showVolume && (
            <YAxis
              yAxisId='volume'
              orientation='right'
              domain={volumeDomain}
              tickLine={false}
              axisLine={false}
              tickMargin={10}
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => {
                if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
                if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
                return value.toString();
              }}
            />
          )}

          <Tooltip
            labelFormatter={(value) => new Date(value).toLocaleString()}
            formatter={(value, name, props) => {
              if (value === null || value === undefined) return null;

              const dataKey = props?.dataKey;

              if (dataKey === "volume") {
                const v = Number(value);
                if (v >= 1000000) return [`${(v / 1000000).toFixed(2)}M`, "Volume"];
                if (v >= 1000) return [`${(v / 1000).toFixed(1)}K`, "Volume"];
                return [v.toString(), "Volume"];
              }

              return [`$${Number(value).toLocaleString()}`, name];
            }}
            contentStyle={{
              backgroundColor: "oklch(0.214 0.009 43.1)",
              border: "1px solid rgba(255,255,255,0.2)",
              borderRadius: "8px",
              color: "#fff"
            }}
          />

          <Legend wrapperStyle={{ paddingTop: "20px" }} iconType='line' />

          <ReferenceLine
            x={predictionStartDate}
            stroke='rgba(255,255,255,0.2)'
            strokeWidth={2}
            strokeDasharray='3 3'
            label={{
              value: "Prediction",
              position: "top",
              fill: "rgba(255,255,255,0.2)",
              fontSize: 10
            }}
          />

          {/* Historical data - solid line */}
          <Line
            type='monotone'
            dataKey='price'
            stroke='#10b981'
            strokeWidth={2.5}
            dot={false}
            name='Historical Price'
            isAnimationActive={false}
          />

          {/* Prediction data - dashed line */}
          <Line
            type='monotone'
            dataKey='prediction'
            stroke='#3b82f6'
            strokeWidth={2.5}
            // strokeDasharray='5 5'
            dot={false}
            connectNulls={true}
            name='Predicted Price'
            isAnimationActive={false}
          />

          {/* Volume bars */}
          {showVolume && (
            <Bar
              dataKey='volume'
              yAxisId='volume'
              fill='rgba(99, 102, 241, 0.5)'
              name='Volume'
              isAnimationActive={false}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PriceChart;
