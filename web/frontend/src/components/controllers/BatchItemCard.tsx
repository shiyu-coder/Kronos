"use client";

import { Input } from "@/components/ui/input";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList
} from "@/components/ui/combobox";
import LabelWithTooltip from "@/components/TooltipWithLabel";
import { DataSourceEnum } from "@/schemas/predictionSchema";
import type { PricePredictionRequest } from "@/schemas/predictionSchema";
import { usePricePredictionStore } from "@/stores/pricePredictionStore";
import { HugeiconsIcon } from "@hugeicons/react";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import z from "zod";

const dataSources = DataSourceEnum.options;

const fieldDescriptions = {
  dataSource: "Select the data source for fetching market data: binance, yfinance, or local CSV file.",
  symbol:
    "Trading pair symbol. For Binance use format like BTCUSDT, ETHUSDT. For yfinance use format like ETH-USD, BTC-USD.",
  period: "Time period for historical data (e.g., 30d, 90d, 1y). Format: number + unit (d=days, y=years).",
  limit:
    "Number of data points to fetch. Used for Binance data source. Example: interval 1h for 1 month data = limit: 720 (24 hours × 30 days)",
  interval: "Candle interval/timeframe (e.g., 15m, 1h, 4h, 1d). Controls data granularity.",
  temperature: "Controls randomness in generation. Higher values make output more random. Set to 0 for deterministic.",
  topP: "Nucleus sampling threshold. Only tokens with cumulative probability above this are considered.",
  topK: "Number of highest probability tokens to keep. 0 disables top-k filtering.",
  sampleCount: "Number of predictions to generate. More samples provide variety but take longer."
};

type BatchItemCardProps = {
  index: number;
  item: Partial<PricePredictionRequest>;
  canRemove: boolean;
};

const BatchItemCard = ({ index, item, canRemove }: BatchItemCardProps) => {
  const updateBatchItem = usePricePredictionStore((state) => state.updateBatchItem);
  const removeBatchItem = usePricePredictionStore((state) => state.removeBatchItem);

  const dataSource = item.data_source ?? "binance";

  return (
    <div className='relative flex flex-col gap-2 border border-muted-foreground/20 p-3 mt-2'>
      {/* Header row */}
      <div className='flex items-center justify-between'>
        <span className='text-xs font-semibold text-muted-foreground uppercase tracking-wider'>
          Symbol #{index + 1}
        </span>
        {canRemove && (
          <button
            type='button'
            className='grid size-6 place-items-center rounded text-muted-foreground hover:text-red-400 hover:bg-red-400/10 transition-colors cursor-pointer'
            onClick={() => removeBatchItem(index)}
          >
            <HugeiconsIcon icon={Delete02Icon} size={14} strokeWidth={2} />
          </button>
        )}
      </div>

      {/* Data source + Symbol */}
      <div className='grid grid-cols-2 gap-2'>
        <div className='grid gap-1'>
          <LabelWithTooltip label='Source' description={fieldDescriptions.dataSource} />
          <Combobox
            items={dataSources}
            value={dataSource}
            onValueChange={(value) => updateBatchItem(index, { data_source: value as z.infer<typeof DataSourceEnum> })}
          >
            <ComboboxInput placeholder='Source' className='uppercase text-xs' style={{ textTransform: "uppercase" }} />
            <ComboboxContent>
              <ComboboxEmpty>No items found.</ComboboxEmpty>
              <ComboboxList>
                {(comboItem) => (
                  <ComboboxItem key={comboItem} value={comboItem} className='uppercase'>
                    {comboItem !== "local" ? comboItem : "Local (CSV)"}
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        </div>

        <div className='grid gap-1'>
          <LabelWithTooltip label='Symbol' description={fieldDescriptions.symbol} />
          <Input
            placeholder='e.g. BTCUSDT'
            className='text-xs'
            value={item.symbol ?? ""}
            onChange={(e) => updateBatchItem(index, { symbol: e.target.value })}
          />
        </div>
      </div>

      {/* Source-specific fields + interval */}
      <div className={`grid ${dataSource === "local" ? "grid-cols-1" : "grid-cols-2"} gap-2`}>
        {dataSource === "binance" && (
          <div className='grid gap-1'>
            <LabelWithTooltip label='Limit' description={fieldDescriptions.limit} />
            <Input
              type='number'
              placeholder='400'
              className='text-xs'
              value={item.limit ?? ""}
              onChange={(e) =>
                updateBatchItem(index, { limit: e.target.value === "" ? undefined : Number(e.target.value) })
              }
            />
          </div>
        )}

        {dataSource === "yfinance" && (
          <div className='grid gap-1'>
            <LabelWithTooltip label='Period' description={fieldDescriptions.period} />
            <Input
              placeholder='30d'
              className='text-xs'
              value={item.period ?? ""}
              onChange={(e) => updateBatchItem(index, { period: e.target.value })}
            />
          </div>
        )}

        {dataSource !== "local" && (
          <div className='grid gap-1'>
            <LabelWithTooltip label='Interval' description={fieldDescriptions.interval} />
            <Input
              placeholder='1h'
              className='text-xs'
              value={item.interval ?? ""}
              onChange={(e) => updateBatchItem(index, { interval: e.target.value })}
            />
          </div>
        )}
      </div>

      {/* Per-item sampling overrides */}
      <div className='grid grid-cols-2 gap-2'>
        <div className='grid gap-1'>
          <LabelWithTooltip label='Temperature' description={fieldDescriptions.temperature} />
          <Input
            type='number'
            placeholder='1.0'
            className='text-xs'
            value={item.temperature ?? ""}
            onChange={(e) =>
              updateBatchItem(index, { temperature: e.target.value === "" ? undefined : Number(e.target.value) })
            }
          />
        </div>
        <div className='grid gap-1'>
          <LabelWithTooltip label='Top P' description={fieldDescriptions.topP} />
          <Input
            type='number'
            placeholder='0.9'
            className='text-xs'
            value={item.top_p ?? ""}
            onChange={(e) =>
              updateBatchItem(index, { top_p: e.target.value === "" ? undefined : Number(e.target.value) })
            }
          />
        </div>
      </div>

      <div className='grid grid-cols-2 gap-2'>
        <div className='grid gap-1'>
          <LabelWithTooltip label='Top K' description={fieldDescriptions.topK} />
          <Input
            type='number'
            placeholder='0'
            className='text-xs'
            value={item.top_k ?? ""}
            onChange={(e) =>
              updateBatchItem(index, { top_k: e.target.value === "" ? undefined : Number(e.target.value) })
            }
          />
        </div>
        <div className='grid gap-1'>
          <LabelWithTooltip label='Samples' description={fieldDescriptions.sampleCount} />
          <Input
            type='number'
            placeholder='1'
            className='text-xs'
            value={item.sample_count ?? ""}
            onChange={(e) =>
              updateBatchItem(index, { sample_count: e.target.value === "" ? undefined : Number(e.target.value) })
            }
          />
        </div>
      </div>
    </div>
  );
};

export default BatchItemCard;
