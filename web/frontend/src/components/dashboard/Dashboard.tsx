"use client";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

import InfoPanel from "./InfoPanel";
import PredictionChart from "./PredictionChart";
import { usePricePredictionStore } from "@/stores/pricePredictionStore";
import { ChartCandlestickIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import WaveLoader from "../WaveLoader";

const Dashboard = () => {
  const mode = usePricePredictionStore((state) => state.mode);
  const result = usePricePredictionStore((state) => state.result);
  const batchResults = usePricePredictionStore((state) => state.batchResults);
  const activeResultIndex = usePricePredictionStore((state) => state.activeResultIndex);
  const setActiveResultIndex = usePricePredictionStore((state) => state.setActiveResultIndex);
  const isLoading = usePricePredictionStore((state) => state.isLoading);

  const hasBatchResults = batchResults && batchResults.length > 0;
  const activeBatchResult = hasBatchResults ? batchResults[activeResultIndex] : null;

  // Determine which result to display
  const displayResult = mode === "batch" ? activeBatchResult : result;
  const hasResult = displayResult !== null;

  return (
    <section className='size-full border'>
      <Card className='size-full'>
        <CardHeader>
          <InfoPanel />
        </CardHeader>

        <CardContent className=''>
          <div className='flex flex-col gap-4'>
            <div className='flex flex-col gap-0'>
              <h2 className='text-base font-semibold'>Prediction preview</h2>
              <p className='text-sm/relaxed text-muted-foreground'>
                History candles transition into the forecast region.
              </p>
            </div>

            {/* Batch result tabs */}
            {mode === "batch" && hasBatchResults && (
              <Tabs
                value={String(activeResultIndex)}
                onValueChange={(value) => setActiveResultIndex(Number(value))}
                className='w-full'
              >
                <TabsList className='w-full flex-wrap h-auto gap-1 p-1'>
                  {batchResults.map((res, index) => (
                    <TabsTrigger
                      key={index}
                      value={String(index)}
                      className='text-xs px-3 py-1.5'
                    >
                      {res.request.symbol || `#${index + 1}`}
                    </TabsTrigger>
                  ))}
                </TabsList>
              </Tabs>
            )}

            {isLoading ? (
              <div className='h-[500px] w-full border flex flex-col items-center justify-center gap-4 animate-pulse'>
                <WaveLoader className='' primaryBgClass='' />
                <p className='text-muted-foreground '>
                  {mode === "batch" ? "Loading batch predictions..." : "Loading prediction..."}
                </p>
              </div>
            ) : hasResult ? (
              <PredictionChart data={displayResult} />
            ) : (
              <div className='h-[500px] w-full border flex flex-col items-center justify-center gap-4'>
                <HugeiconsIcon
                  icon={ChartCandlestickIcon}
                  size={48}
                  strokeWidth={1.5}
                  className='text-muted-foreground'
                />
                <div className='text-center'>
                  <p className='text-muted-foreground'>No prediction data yet</p>
                  <p className='text-sm text-muted-foreground/60'>Run a prediction to see the chart</p>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </section>
  );
};

export default Dashboard;
