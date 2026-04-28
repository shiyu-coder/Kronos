"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { SaveIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { saveResult } from "@/lib/api/savedResults";
import type { PricePredictionBatchResponse, PricePredictionResponse } from "@/schemas/predictionSchema";
import type { SaveResultRequest } from "@/schemas/resultsSchema";

type SaveResultActionProps = {
  result: PricePredictionResponse;
  batchResults: PricePredictionBatchResponse | null;
  isBatch: boolean;
};

function buildLabel(result: PricePredictionResponse, isBatch: boolean, batchCount: number) {
  const symbol = result.request.symbol || "Local data";
  const date = new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date());

  if (isBatch) {
    return `Batch ${batchCount} results - ${date}`;
  }

  return `${symbol} ${result.request.interval} - ${date}`;
}

export default function SaveResultAction({ result, batchResults, isBatch }: SaveResultActionProps) {
  const queryClient = useQueryClient();
  const batchCount = batchResults?.length ?? 0;

  const mutation = useMutation({
    mutationFn: () => {
      const body: SaveResultRequest =
        isBatch && batchResults && batchResults.length > 0
          ? {
              type: "predict/price/batch",
              label: buildLabel(result, true, batchResults.length),
              data: batchResults
            }
          : {
              type: "predict/price",
              label: buildLabel(result, false, 1),
              data: result
            };

      return saveResult(body);
    },
    onSuccess: (saved) => {
      queryClient.invalidateQueries({ queryKey: ["saved-results"] });
      toast.success("Prediction saved", {
        description: saved.label || "Saved result is ready in Saved Results."
      });
    },
    onError: (error) => {
      toast.error("Failed to save prediction", {
        description: error instanceof Error ? error.message : "Please try again."
      });
    }
  });

  const tooltipLabel = isBatch
    ? "Save all batch predictions"
    : "Save this prediction";

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            size="sm"
            className="h-8 gap-1.5 text-xs px-2.5 cursor-pointer" 
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || (isBatch && batchCount === 0)}
            id="save-prediction-result"
          >
            <HugeiconsIcon icon={SaveIcon} />
            {mutation.isPending ? "Saving" : "Save"}
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          <p className="text-xs">{tooltipLabel}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
