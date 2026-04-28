"use client";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Combobox, ComboboxInput, ComboboxContent, ComboboxList, ComboboxItem } from "@/components/ui/combobox";

import InfoPanel from "./InfoPanel";
import PredictionChart from "./PredictionChart";
import { usePricePredictionStore } from "@/stores/pricePredictionStore";
import { ChartCandlestickIcon, Alert02Icon, CancelCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import WaveLoader from "../WaveLoader";

const Dashboard = () => {
  const mode = usePricePredictionStore((state) => state.mode);
  const result = usePricePredictionStore((state) => state.result);
  const batchResults = usePricePredictionStore((state) => state.batchResults);
  const activeResultIndex = usePricePredictionStore((state) => state.activeResultIndex);
  const setActiveResultIndex = usePricePredictionStore((state) => state.setActiveResultIndex);
  const isLoading = usePricePredictionStore((state) => state.isLoading);
  const error = usePricePredictionStore((state) => state.error);
  const clearError = usePricePredictionStore((state) => state.clearError);

  const hasBatchResults = batchResults && batchResults.length > 0;
  const activeBatchResult = hasBatchResults ? batchResults[activeResultIndex] : null;

  // Determine which result to display
  const displayResult = mode === "batch" ? activeBatchResult : result;
  const hasResult = displayResult !== null;

  // Build combobox options from batch results
  const batchOptions = hasBatchResults
    ? batchResults.map((res, index) => {
        const symbol = res.request.symbol || `Item`;
        const label = `#${index + 1} — ${symbol}`;
        const detail = `${res.request.interval} · ${res.request.data_source}`;
        const dataSource = res.request.data_source || "";
        return { value: String(index), label, detail, dataSource };
      })
    : [];

  const activeOption = batchOptions[activeResultIndex];

  return (
    <section className='size-full border'>
      <Card className='size-full'>
        <CardHeader>
          <InfoPanel />
        </CardHeader>

        <CardContent className='flex-1'>
          <div className='flex flex-col gap-4 h-full'>
            <div className='flex flex-col gap-0'>
              <h2 className='text-base font-semibold'>Prediction preview</h2>
              <p className='text-sm/relaxed text-muted-foreground'>
                History candles transition into the forecast region.
              </p>
            </div>

            {hasResult && (
              <div className='grid grid-cols-2 gap-4'>
                {/* Batch result combobox */}
                {mode === "batch" && hasBatchResults && (
                  <Combobox
                    value={activeOption ? `${activeOption.label} (${activeOption.dataSource.toUpperCase()})` : ""}
                    onValueChange={(val) => {
                      const match = batchOptions.find((opt) => opt.label === val);
                      if (match) setActiveResultIndex(Number(match.value));
                    }}
                  >
                    <ComboboxInput placeholder='Switch batch result…' />
                    <ComboboxContent>
                      <ComboboxList>
                        {batchOptions.map((opt) => (
                          <ComboboxItem key={opt.value} value={opt.label}>
                            <div className='flex flex-col gap-0.5'>
                              <span className='font-medium'>{opt.label}</span>
                              <span className='text-[11px] text-muted-foreground uppercase'>{opt.detail}</span>
                            </div>
                          </ComboboxItem>
                        ))}
                      </ComboboxList>
                    </ComboboxContent>
                  </Combobox>
                )}
              </div>
            )}

            {isLoading ? (
              <div className='size-full border flex flex-col items-center justify-center gap-4 animate-pulse'>
                <WaveLoader className='' primaryBgClass='' />
                <p className='text-muted-foreground '>
                  {mode === "batch" ? "Loading batch predictions..." : "Loading prediction..."}
                </p>
              </div>
            ) : error ? (
              <div className='size-full border border-red-500/30 bg-red-500/5 flex flex-col items-center justify-center gap-5 relative'>
                {/* Dismiss button */}
                <Button
                  variant='ghost'
                  size='sm'
                  onClick={clearError}
                  className='absolute top-3 right-3 h-7 w-7 p-0 text-red-400 hover:text-red-300 hover:bg-red-500/10'
                >
                  <HugeiconsIcon icon={CancelCircleIcon} size={16} strokeWidth={2} />
                </Button>

                {/* Error icon */}
                <div className='size-14 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center'>
                  <HugeiconsIcon icon={Alert02Icon} size={28} strokeWidth={1.5} className='text-red-400' />
                </div>

                {/* Error content */}
                <div className='text-center max-w-md px-4 flex flex-col gap-2'>
                  <p className='text-sm font-semibold text-red-400'>{error.title}</p>
                  <p className='text-sm text-red-300/80 leading-relaxed'>{error.message}</p>
                  {error.code && (
                    <span className='inline-block mt-1 text-xs font-mono text-red-400/60 bg-red-500/10 border border-red-500/20 px-2 py-0.5 rounded w-fit mx-auto'>
                      {error.code}
                    </span>
                  )}
                </div>

                {/* Timestamp */}
                <p className='text-xs text-red-400/40 font-mono'>{error.timestamp.toLocaleTimeString()}</p>
              </div>
            ) : hasResult ? (
              <PredictionChart data={displayResult} />
            ) : (
              <div className='size-full border flex flex-col items-center justify-center gap-4'>
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
