"use client";
import { useState } from "react";

import { useMutation } from "@tanstack/react-query";
import * as Dialog from "@radix-ui/react-dialog";
import { XIcon } from "@phosphor-icons/react";
import { PlayIcon, RefreshIcon, Setting07Icon } from "@hugeicons/core-free-icons";
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

import DataSelector from "./DataSelector";
import ModelSelector from "./ModelSelector";

import { usePricePredictionStore } from "@/stores/pricePredictionStore";
import { predictPrice, predictPriceDemo } from "@/lib/api";
import { type PricePredictionResponse } from "@/schemas/predictionSchema";

const Controllers = () => {
  const params = usePricePredictionStore((state) => state.params);
  const reset = usePricePredictionStore((state) => state.reset);
  const [result, setResult] = useState<PricePredictionResponse | null>(null);

  const mutation = useMutation({
    mutationFn: predictPriceDemo,
    onSuccess: (data) => {
      setResult(data);
    },
    onError: (error) => {
      console.error("Prediction error:", error);
    }
  });

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
        <div className='w-full flex flex-col gap-2'>
          <h2 className='text-sm font-semibold'>Model choices</h2>
          <ModelSelector />
        </div>

        <DataSelector />
      </CardContent>

      <CardFooter className='size-full flex gap-2 items-end'>
        <Button type='submit' className='flex-2' onClick={() => mutation.mutate(params)} disabled={mutation.isPending}>
          <HugeiconsIcon icon={PlayIcon} size={16} strokeWidth={2} />
          {mutation.isPending ? "Predicting..." : "Predict"}
        </Button>
        <Button variant='outline' className='flex-1' onClick={reset}>
          <HugeiconsIcon icon={RefreshIcon} size={16} strokeWidth={2} />
          Reset
        </Button>
      </CardFooter>

      <Dialog.Root open={!!result} onOpenChange={(open) => !open && setResult(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className='fixed inset-0 bg-black/50 z-50' />
          <Dialog.Content className='fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-background rounded-lg border p-6 shadow-lg z-50 max-w-lg w-full max-h-[80vh] overflow-auto'>
            <div className='flex items-center justify-between mb-4'>
              <Dialog.Title className='text-lg font-semibold'>Prediction Result</Dialog.Title>
              <Dialog.Close asChild>
                <Button variant='ghost' size='icon-xs'>
                  <XIcon />
                </Button>
              </Dialog.Close>
            </div>
            {result && (
              <div className='grid gap-2 text-sm'>
                <p>
                  <strong>Model:</strong> {result.model.name}
                </p>
                <p>
                  <strong>Symbol:</strong> {result.request.symbol}
                </p>
                <p>
                  <strong>Lookback:</strong> {result.request.lookback}
                </p>
                <p>
                  <strong>Pred Length:</strong> {result.request.pred_len}
                </p>
                <p>
                  <strong>History records:</strong> {result.history.length}
                </p>
                <p>
                  <strong>Prediction records:</strong> {result.prediction.length}
                </p>
              </div>
            )}
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </Card>
  );
};

export default Controllers;
