"use client";

import { useRef, useState } from "react";
import { AlertCircleIcon, ChartCandlestickIcon, DatabaseIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import PredictionChart from "@/components/dashboard/PredictionChart";
import DownloadActions from "@/components/dashboard/DownloadActions";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Combobox, ComboboxContent, ComboboxInput, ComboboxItem, ComboboxList } from "@/components/ui/combobox";
import type { PricePredictionResponse } from "@/schemas/predictionSchema";
import type { SavedResultDetail } from "@/schemas/resultsSchema";

type SavedResultViewerProps = {
  detail: SavedResultDetail | undefined;
  isLoading: boolean;
  isError: boolean;
  onRetry: () => void;
};

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(value));
}

function isBatchData(data: SavedResultDetail["data"]): data is PricePredictionResponse[] {
  return Array.isArray(data);
}

function Metric({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <Card size='sm' className='gap-1 p-3'>
      <CardDescription>{label}</CardDescription>
      <CardTitle className='truncate text-base'>{value ?? "N/A"}</CardTitle>
    </Card>
  );
}

function CandleTable({ result }: { result: PricePredictionResponse }) {
  const rows = result.prediction.slice(0, 10);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction Data</CardTitle>
        <CardDescription>First {rows.length} forecast candles from the saved result.</CardDescription>
      </CardHeader>
      <CardContent className='overflow-x-auto'>
        <table className='w-full min-w-[680px] border-collapse text-xs'>
          <thead className='text-muted-foreground'>
            <tr className='border-b'>
              <th className='py-2 pr-3 text-left font-medium'>Timestamp</th>
              <th className='py-2 pr-3 text-right font-medium'>Open</th>
              <th className='py-2 pr-3 text-right font-medium'>High</th>
              <th className='py-2 pr-3 text-right font-medium'>Low</th>
              <th className='py-2 pr-3 text-right font-medium'>Close</th>
              <th className='py-2 text-right font-medium'>Volume</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.timestamps} className='border-b last:border-b-0'>
                <td className='py-2 pr-3 text-muted-foreground'>{row.timestamps}</td>
                <td className='py-2 pr-3 text-right'>{row.open.toFixed(4)}</td>
                <td className='py-2 pr-3 text-right'>{row.high.toFixed(4)}</td>
                <td className='py-2 pr-3 text-right'>{row.low.toFixed(4)}</td>
                <td className='py-2 pr-3 text-right'>{row.close.toFixed(4)}</td>
                <td className='py-2 text-right'>{row.volume.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

export default function SavedResultViewer({ detail, isLoading, isError, onRetry }: SavedResultViewerProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const [batchIndex, setBatchIndex] = useState(0);

  if (isLoading) {
    return (
      <div className='flex h-full flex-col gap-4 p-6'>
        <div className='h-24 animate-pulse border bg-muted/20' />
        <div className='h-[520px] animate-pulse border bg-muted/20' />
      </div>
    );
  }

  if (isError) {
    return (
      <div className='flex h-full flex-col items-center justify-center gap-4 p-6 text-center'>
        <HugeiconsIcon icon={AlertCircleIcon} className='size-10 text-destructive' />
        <div>
          <p className='text-sm font-semibold'>Unable to load this saved result</p>
          <p className='text-muted-foreground'>The record may have been deleted or the API returned an error.</p>
        </div>
        <Button variant='outline' onClick={onRetry}>
          Retry
        </Button>
      </div>
    );
  }

  if (!detail) {
    return (
      <div className='flex h-full flex-col items-center justify-center gap-4 p-6 text-center'>
        <HugeiconsIcon icon={ChartCandlestickIcon} className='size-12 text-muted-foreground' />
        <div>
          <p className='text-sm font-semibold'>Select a saved prediction</p>
          <p className='text-muted-foreground'>Choose a result from the sidebar to view its chart and data.</p>
        </div>
      </div>
    );
  }

  const data = detail.data;
  const isBatch = isBatchData(data);
  if (isBatch && data.length === 0) {
    return (
      <div className='flex h-full flex-col items-center justify-center gap-4 p-6 text-center'>
        <HugeiconsIcon icon={ChartCandlestickIcon} className='size-12 text-muted-foreground' />
        <div>
          <p className='text-sm font-semibold'>Empty saved batch</p>
          <p className='text-muted-foreground'>This saved result does not contain any prediction items.</p>
        </div>
      </div>
    );
  }

  const batchResults: PricePredictionResponse[] | null = isBatch ? data : null;
  const activeResult: PricePredictionResponse = isBatch ? data[Math.min(batchIndex, data.length - 1)] : data;
  const batchOptions =
    batchResults?.map((result, index) => ({
      value: String(index),
      label: `#${index + 1} ${result.request.symbol || "Item"}`,
      detail: `${result.request.interval} / ${result.request.data_source}`
    })) ?? [];
  const activeOption = batchOptions[batchIndex];

  return (
    <div className='flex min-h-screen flex-col gap-4 px-4 sm:x-0'>
      <header className='flex flex-col gap-4 border-b pb-4 lg:flex-row lg:items-start lg:justify-between'>
        <div className='min-w-0'>
          <p className='text-[11px] font-bold uppercase tracking-[0.18em] text-muted-foreground'>
            {isBatch ? "Saved batch result" : "Saved result"}
          </p>
          <h2 className='truncate text-xl font-semibold'>{detail.label || "Untitled prediction"}</h2>
          <p className='text-xs text-muted-foreground'>Created {formatDate(detail.created_at)}</p>
        </div>

        <DownloadActions
          chartRef={chartRef}
          displayResult={activeResult}
          batchResults={batchResults}
          isBatch={isBatch}
          captureChart={async (index) => {
            setBatchIndex(index);
            await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
            return chartRef.current;
          }}
        />
      </header>

      <section className='grid grid-cols-1 gap-3 lg:grid-cols-4'>
        <Metric label='Market' value={activeResult.request.symbol || "Local data"} />
        <Metric label='Model' value={activeResult.request.model_name} />
        <Metric label='Interval' value={`${activeResult.request.interval} / ${activeResult.request.period}`} />
        <Metric label='Forecast' value={`${activeResult.request.pred_len} steps`} />
      </section>

      {isBatch && batchOptions.length > 0 && (
        <div className='max-w-md bg-primary-foreground'>
          <Combobox
            value={activeOption ? `${activeOption.label} (${activeOption.detail})` : ""}
            onValueChange={(value) => {
              const match = batchOptions.find((option) => option.label === value);
              if (match) setBatchIndex(Number(match.value));
            }}
          >
            <ComboboxInput placeholder='Switch batch result...' />
            <ComboboxContent>
              <ComboboxList>
                {batchOptions.map((option) => (
                  <ComboboxItem key={option.value} value={option.label}>
                    <div className='flex flex-col gap-0.5'>
                      <span className='font-medium'>{option.label}</span>
                      <span className='text-[11px] uppercase text-muted-foreground'>{option.detail}</span>
                    </div>
                  </ComboboxItem>
                ))}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        </div>
      )}

      <Card className='min-h-[560px]'>
        <CardHeader className='flex-row items-start justify-between gap-4'>
          <div>
            <CardTitle>Chart</CardTitle>
            <CardDescription>Historical candles with the saved forecast region.</CardDescription>
          </div>
          <div className='flex items-center gap-2 text-xs text-muted-foreground'>
            <HugeiconsIcon icon={DatabaseIcon} className='size-4' />
            <span>
              {activeResult.history.length} history / {activeResult.prediction.length} prediction
            </span>
          </div>
        </CardHeader>
        <CardContent>
          <PredictionChart ref={chartRef} data={activeResult} />
        </CardContent>
      </Card>

      <CandleTable result={activeResult} />
    </div>
  );
}
