"use client";

import { toast } from "sonner";

import { useMutation } from "@tanstack/react-query";
import { PlayIcon, RefreshIcon, Setting07Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

import DataSelector from "./DataSelector";
import ModelSelector from "./ModelSelector";
import BatchItemCard from "./BatchItemCard";

import { usePricePredictionStore, type PredictionMode } from "@/stores/pricePredictionStore";
import { predictPrice, predictPriceBatch } from "@/lib/api";
import { PredictionApiError } from "@/utils/handleAPI";
import type { PricePredictionRequest } from "@/schemas/predictionSchema";

const MAX_BATCH_SIZE = 20;

const Controllers = () => {
  const params = usePricePredictionStore((state) => state.params);
  const batchParams = usePricePredictionStore((state) => state.batchParams);
  const mode = usePricePredictionStore((state) => state.mode);
  const setMode = usePricePredictionStore((state) => state.setMode);
  const reset = usePricePredictionStore((state) => state.reset);
  const setResult = usePricePredictionStore((state) => state.setResult);
  const setBatchResults = usePricePredictionStore((state) => state.setBatchResults);
  const setIsLoading = usePricePredictionStore((state) => state.setIsLoading);
  const setError = usePricePredictionStore((state) => state.setError);
  const clearError = usePricePredictionStore((state) => state.clearError);
  const batchItems = usePricePredictionStore((state) => state.batchItems);
  const addBatchItem = usePricePredictionStore((state) => state.addBatchItem);
  const setActiveResultIndex = usePricePredictionStore((state) => state.setActiveResultIndex);

  // --- Single prediction mutation ---
  const predictPriceMutation = useMutation({
    mutationFn: predictPrice,
    onSuccess: (data) => {
      setResult(data);
      clearError();
      setIsLoading(false);
    },
    onError: (error) => {
      console.error("Prediction error:", error);
      const apiErr = error instanceof PredictionApiError ? error : null;
      const errTitle = apiErr?.code ? `${apiErr.code} (${apiErr.status})` : `Prediction Failed`;
      const errMessage = apiErr?.message ?? error.message ?? "An unexpected error occurred.";

      setError({ title: errTitle, message: errMessage, code: apiErr?.code, timestamp: new Date() });
      toast.error(errTitle, {
        description: errMessage,
        duration: 6000
      });

      setIsLoading(false);
      setResult(null);
    },
    onMutate: () => {
      setIsLoading(true);
      setResult(null);
      clearError();
    }
  });

  // --- Batch prediction mutation ---
  const predictPriceBatchMutation = useMutation({
    mutationFn: predictPriceBatch,
    onSuccess: (data) => {
      setBatchResults(data);
      setActiveResultIndex(0);
      clearError();
      setIsLoading(false);
    },
    onError: (error) => {
      console.error("Batch prediction error:", error);
      const apiErr = error instanceof PredictionApiError ? error : null;
      const errTitle = apiErr?.code ? `${apiErr.code} (${apiErr.status})` : `Batch Prediction Failed`;
      const errMessage = apiErr?.message ?? error.message ?? "An unexpected error occurred.";

      setError({ title: errTitle, message: errMessage, code: apiErr?.code, timestamp: new Date() });
      toast.error(errTitle, {
        description: errMessage,
        duration: 6000
      });

      setIsLoading(false);
      setBatchResults(null);
    },
    onMutate: () => {
      setIsLoading(true);
      setBatchResults(null);
      clearError();
    }
  });

  const handlePredict = () => {
    if (mode === "single") {
      predictPriceMutation.mutate(params);
    } else {
      // Merge batch shared params with each batch item's per-item overrides

      const batchRequests: PricePredictionRequest[] = batchItems.map((item) => ({
        ...batchParams, // batch shared: model_name, device, lookback, pred_len
        ...item // per-item overrides: data_source, symbol, interval, period, limit, sampling params
      }));

      predictPriceBatchMutation.mutate(batchRequests);
    }
  };

  const handleReset = () => {
    reset();
    setResult(null);
    setBatchResults(null);
    setIsLoading(false);
  };

  const isPending = mode === "single" ? predictPriceMutation.isPending : predictPriceBatchMutation.isPending;

  return (
    <Card className='flex size-full'>
      <CardHeader>
        <CardTitle>Run setup</CardTitle>
        <CardDescription>Configure source, model, and sampling.</CardDescription>
        <CardAction>
          <HugeiconsIcon icon={Setting07Icon} size={22} strokeWidth={1.8} />
        </CardAction>
      </CardHeader>

      <CardContent className='flex flex-col gap-3'>
        {/* Mode toggle */}
        <Tabs value={mode} onValueChange={(value) => setMode(value as PredictionMode)} className='w-full'>
          <TabsList className='grid w-full grid-cols-2'>
            <TabsTrigger value='single'>Single</TabsTrigger>
            <TabsTrigger value='batch'>Batch</TabsTrigger>
          </TabsList>
        </Tabs>

        {/* Model selector (always visible) */}
        <div className='w-full flex flex-col gap-2'>
          <h2 className='text-sm font-semibold'>Model choices</h2>
          <ModelSelector />
        </div>

        <>
          {/* Shared parameters (always visible) */}
          <DataSelector />
          {/* Batch items list (batch mode only) */}
          {mode === "batch" && (
            <div className='flex flex-col gap-2 '>
              <div className='flex items-center justify-between'>
                <h2 className='text-sm font-semibold'>
                  Symbols
                  <span className='text-xs font-normal text-muted-foreground'>
                    ({batchItems.length}/{MAX_BATCH_SIZE})
                  </span>
                </h2>
                <Button
                  variant='outline'
                  size='sm'
                  onClick={addBatchItem}
                  disabled={batchItems.length >= MAX_BATCH_SIZE}
                  className='h-7 text-xs gap-1'
                >
                  <HugeiconsIcon icon={PlusSignIcon} size={12} strokeWidth={2} />
                  Add
                </Button>
              </div>

              <div className='flex flex-col gap-2 max-h-[400px] overflow-y-auto py-1 pr-2'>
                {batchItems.map((item, index) => (
                  <BatchItemCard key={index} index={index} item={item} canRemove={batchItems.length > 1} />
                ))}
              </div>
            </div>
          )}
        </>
      </CardContent>

      <CardFooter className='size-full flex gap-2 items-end'>
        <Button type='submit' className='flex-2' onClick={handlePredict} disabled={isPending}>
          <HugeiconsIcon icon={PlayIcon} size={16} strokeWidth={2} />
          {isPending ? "Predicting..." : mode === "batch" ? `Predict Batch (${batchItems.length})` : "Predict"}
        </Button>
        <Button variant='outline' className='flex-1' onClick={handleReset}>
          <HugeiconsIcon icon={RefreshIcon} size={16} strokeWidth={2} />
          Reset
        </Button>
      </CardFooter>
    </Card>
  );
};

export default Controllers;
