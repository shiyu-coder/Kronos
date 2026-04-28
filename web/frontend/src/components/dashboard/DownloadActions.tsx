"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Image02Icon, Download01Icon, ArchiveIcon, Svg01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import type { PricePredictionResponse } from "@/schemas/predictionSchema";
import {
  downloadChartAsPNG,
  downloadChartAsSVG,
  downloadResultAsCSV,
  downloadBatchAsZip,
  buildFilenamePrefix
} from "@/utils/downloadUtils";

type DownloadActionsProps = {
  /** Ref to the chart container DOM node (for screenshot capture) */
  chartRef: React.RefObject<HTMLDivElement | null>;
  /** The currently displayed prediction result */
  displayResult: PricePredictionResponse;
  /** All batch results (null/empty when in single mode) */
  batchResults?: PricePredictionResponse[] | null;
  /** Whether we're in batch mode */
  isBatch: boolean;
  /**
   * For "Download All" in batch mode: called for each index i.
   * Caller switches the chart to index i, waits for repaint, returns the
   * live chart DOM node. Return null to skip chart capture for that index.
   */
  captureChart?: (index: number) => Promise<HTMLElement | null>;
};

const DownloadActions = ({
  chartRef,
  displayResult,
  batchResults,
  isBatch,
  captureChart
}: DownloadActionsProps) => {
  const [busy, setBusy] = useState(false);
  const prefix = buildFilenamePrefix(displayResult);

  const handlePNG = async () => {
    if (!chartRef.current) return;
    try {
      setBusy(true);
      await downloadChartAsPNG(chartRef.current, `${prefix}.png`);
      toast.success("Chart downloaded", { description: `${prefix}.png` });
    } catch (e) {
      console.error(e);
      toast.error("Failed to download chart as PNG");
    } finally {
      setBusy(false);
    }
  };

  const handleSVG = () => {
    if (!chartRef.current) return;
    try {
      downloadChartAsSVG(chartRef.current, `${prefix}.svg`);
      toast.success("Chart downloaded", { description: `${prefix}.svg` });
    } catch (e) {
      console.error(e);
      toast.error("Failed to download chart as SVG");
    }
  };

  const handleCSV = () => {
    try {
      downloadResultAsCSV(displayResult, `${prefix}.csv`);
      toast.success("Data downloaded", { description: `${prefix}.csv` });
    } catch (e) {
      console.error(e);
      toast.error("Failed to download data as CSV");
    }
  };

  const handleBatchZip = async () => {
    if (!batchResults || batchResults.length === 0) return;
    try {
      setBusy(true);
      toast.info("Preparing batch download…", { description: "This may take a moment." });

      await downloadBatchAsZip(
        batchResults,
        captureChart ?? (() => Promise.resolve(null))
      );

      toast.success("Batch download complete", {
        description: `${batchResults.length} results exported as ZIP`
      });
    } catch (e) {
      console.error(e);
      toast.error("Failed to create batch ZIP");
    } finally {
      setBusy(false);
    }
  };

  return (
    <TooltipProvider delayDuration={200}>
      <div className='flex items-center gap-1'>
        {/* Download as PNG */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant='outline'
              size='sm'
              className='h-8 w-8 p-0 cursor-pointer'
              onClick={handlePNG}
              disabled={busy}
              id='download-chart-png'
            >
              <HugeiconsIcon icon={Image02Icon} size={14} strokeWidth={1.8} />
            </Button>
          </TooltipTrigger>
          <TooltipContent side='bottom'>
            <p className='text-xs'>Download chart as PNG</p>
          </TooltipContent>
        </Tooltip>

        {/* Download as SVG */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant='outline'
              size='sm'
              className='h-8 w-8 p-0 cursor-pointer'
              onClick={handleSVG}
              disabled={busy}
              id='download-chart-svg'
            >
              <HugeiconsIcon icon={Svg01Icon} size={14} strokeWidth={1.8} />
            </Button>
          </TooltipTrigger>
          <TooltipContent side='bottom'>
            <p className='text-xs'>Download chart as SVG</p>
          </TooltipContent>
        </Tooltip>

        {/* Download CSV */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant='outline'
              size='sm'
              className='h-8 w-8 p-0 cursor-pointer'
              onClick={handleCSV}
              disabled={busy}
              id='download-data-csv'
            >
              <HugeiconsIcon icon={Download01Icon} size={14} strokeWidth={1.8} />
            </Button>
          </TooltipTrigger>
          <TooltipContent side='bottom'>
            <p className='text-xs'>Download data as CSV</p>
          </TooltipContent>
        </Tooltip>

        {/* Batch ZIP (batch mode only) */}
        {isBatch && batchResults && batchResults.length > 0 && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant='outline'
                size='sm'
                className='h-8 gap-1.5 text-xs px-2.5 cursor-pointer'
                onClick={handleBatchZip}
                disabled={busy}
                id='download-batch-zip'
              >
                <HugeiconsIcon icon={ArchiveIcon} size={14} strokeWidth={1.8} />
                Download All
              </Button>
            </TooltipTrigger>
            <TooltipContent side='bottom'>
              <p className='text-xs'>Download all batch results as ZIP (PNG + SVG + CSV)</p>
            </TooltipContent>
          </Tooltip>
        )}
      </div>
    </TooltipProvider>
  );
};

export default DownloadActions;
