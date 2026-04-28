"use client";

import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  ReferenceLine
} from "recharts";
import { type PricePredictionResponse } from "@/schemas/predictionSchema";

type PriceChartProps = {
  data: PricePredictionResponse | null;
};

const PriceChart = ({ data }: PriceChartProps) => {
  if (!data) {
    return null;
  }

  const historyData = data.history.map((item) => ({
    timestamp: item.timestamps,
    price: item.close,
    prediction: null,
    type: "history"
  }));

  const predictionData = data.prediction.map((item) => ({
    timestamp: item.timestamps,
    price: null,
    prediction: item.close,
    type: "prediction"
  }));

  const allData = [...historyData, ...predictionData];
  // Calculate dynamic Y-axis domain based on data
  const allPrices = allData.map((d) => d.price ?? d.prediction).filter((p) => p !== null) as number[];

  const minPrice = Math.min(...allPrices);
  const maxPrice = Math.max(...allPrices);

  const padding = (maxPrice - minPrice) * 0.1; // 10% padding

  // const yMin = Math.floor((minPrice - padding) / 1000) * 1200;
  // const yMax = Math.ceil((maxPrice + padding) / 1000) * 100;

  const yMin = Math.floor(minPrice - padding);
  const yMax = Math.ceil(maxPrice + padding);

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

          <YAxis
            domain={[yMin, yMax]}
            tickLine={false}
            axisLine={false}
            tickMargin={10}
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => `$${value.toLocaleString()}`}
          />

          <Tooltip
            labelFormatter={(value) => new Date(value).toLocaleString()}
            formatter={(value) => {
              if (value === null || value === undefined) return null;
              return [`$${Number(value).toLocaleString()}`, "Price"];
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
            strokeWidth={1}
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
            isAnimationActive={true}
          />

          {/* Prediction data - dashed line */}
          <Line
            type='monotone'
            dataKey='prediction'
            stroke='#3b82f6'
            strokeWidth={2.5}
            // strokeDasharray='5 5'
            dot={false}
            name='Predicted Price'
            isAnimationActive={true}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PriceChart;
