import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import LabelWithTooltip from "@/components/TooltipWithLabel";

import { DataSourceEnum, DeviceEnum } from "@/schemas/predictionSchema";
import { usePricePredictionStore } from "@/stores/pricePredictionStore";
import z from "zod";

const dataSources = DataSourceEnum.options;
const devices = DeviceEnum.options;

const fieldDescriptions = {
  dataSource: "Select the data source for fetching market data: binance, yfinance, or local CSV file.",
  symbol:
    "Trading pair symbol. For Binance use format like BTCUSDT, ETHUSDT. For yfinance use format like ETH-USD, BTC-USD.",
  lookback:
    "Number of historical candles to use for prediction. Higher values provide more context but require more data.",
  predLen: "Number of future candles to predict. Determines how far ahead the prediction extends.",
  period: "Time period for historical data (e.g., 30d, 90d, 1y). Format: number + unit (d=days, y=years).",
  limit:
    "Number of data points to fetch. Used for Binance data source. Example: interval 1h for 1 month data = limit: 720 (24 hours × 30 days)",
  interval: "Candle interval/timeframe (e.g., 15m, 1h, 4h, 1d). Controls data granularity.",
  device: "Compute device for model inference: CPU, CUDA (NVIDIA GPU), or MPS (Apple Silicon).",
  temperature: "Controls randomness in generation. Higher values make output more random. Set to 0 for deterministic.",
  topP: "Nucleus sampling threshold. Only tokens with cumulative probability above this are considered. Lower = more focused.",
  topK: "Number of highest probability tokens to keep. 0 disables top-k filtering. Limits vocabulary during generation.",
  sampleCount: "Number of predictions to generate. More samples provide variety but take longer to generate."
};

const DataSelector = () => {
  const singleParams = usePricePredictionStore((state) => state.params);
  const setSingleParams = usePricePredictionStore((state) => state.setParams);
  const batchParams = usePricePredictionStore((state) => state.batchParams);
  const setBatchParams = usePricePredictionStore((state) => state.setBatchParams);
  const mode = usePricePredictionStore((state) => state.mode);

  const isSingle = mode === "single";
  const params = isSingle ? singleParams : batchParams;
  const setParams = isSingle ? setSingleParams : setBatchParams;

  return (
    <div className='grid gap-3'>
      {/* Shared Parameters divider — visible in batch mode */}
      {mode === "batch" && (
        <div className='flex items-center gap-2 pt-1'>
          <div className='h-px flex-1 bg-muted-foreground/20' />
          <span className='text-xs font-medium text-muted-foreground uppercase tracking-wider'>Shared Parameters</span>
          <div className='h-px flex-1 bg-muted-foreground/20' />
        </div>
      )}

      {/* Device — always visible (shared) */}
      <div className='grid gap-1'>
        <LabelWithTooltip label='Device' description={fieldDescriptions.device} />
        <Combobox
          items={devices}
          value={params.device}
          onValueChange={(value) => setParams({ device: value as z.infer<typeof DeviceEnum> })}
        >
          <ComboboxInput placeholder='Select device' className='uppercase' style={{ textTransform: "uppercase" }} />
          <ComboboxContent>
            <ComboboxEmpty>No items found.</ComboboxEmpty>
            <ComboboxList>
              {(item) => (
                <ComboboxItem key={item} value={item} className='uppercase'>
                  {item}
                </ComboboxItem>
              )}
            </ComboboxList>
          </ComboboxContent>
        </Combobox>
      </div>

      <div className='grid grid-cols-2 gap-2'>
        <div className='grid gap-2'>
          <LabelWithTooltip label='Lookback' description={fieldDescriptions.lookback} />
          <Input
            type='number'
            placeholder='Enter lookback'
            value={params.lookback ?? ""}
            onChange={(e) => setParams({ lookback: e.target.value === "" ? undefined : Number(e.target.value) })}
          />
        </div>
        <div className='grid gap-2'>
          <LabelWithTooltip label='Predict Length' description={fieldDescriptions.predLen} />
          <Input
            type='number'
            placeholder='Enter predict length'
            value={params.pred_len ?? ""}
            onChange={(e) => setParams({ pred_len: e.target.value === "" ? undefined : Number(e.target.value) })}
          />
        </div>
      </div>

      {/* --- Per-item fields: only in Single mode --- */}
      {isSingle && (
        <>
          <div className='grid gap-1'>
            <LabelWithTooltip label='Data Source' description={fieldDescriptions.dataSource} />
            <Combobox
              items={dataSources}
              value={params.data_source}
              onValueChange={(value) => setParams({ data_source: value as z.infer<typeof DataSourceEnum> })}
            >
              <ComboboxInput placeholder='Select a data' className='uppercase' style={{ textTransform: "uppercase" }} />
              <ComboboxContent>
                <ComboboxEmpty>No items found.</ComboboxEmpty>
                <ComboboxList>
                  {(item) => (
                    <ComboboxItem key={item} value={item} className='uppercase'>
                      {item !== "local" ? item : "Local File (CSV)"}
                    </ComboboxItem>
                  )}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
          </div>

          <div className='grid gap-2'>
            <LabelWithTooltip label='Symbol' description={fieldDescriptions.symbol} />
            <Input
              placeholder='Enter symbol'
              value={params.symbol ?? ""}
              onChange={(e) => setParams({ symbol: e.target.value })}
            />
          </div>

          <div className={`grid ${params.data_source === "local" ? "grid-cols-1" : "grid-cols-2"} gap-2`}>
            {params.data_source === "binance" && (
              <div className='grid gap-2'>
                <LabelWithTooltip label='Limit' description={fieldDescriptions.limit} />
                <Input
                  type='number'
                  placeholder='Enter limit'
                  value={params.limit ?? ""}
                  onChange={(e) => setParams({ limit: e.target.value === "" ? undefined : Number(e.target.value) })}
                />
              </div>
            )}

            {params.data_source === "yfinance" && (
              <div className='grid gap-2'>
                <LabelWithTooltip label='Period' description={fieldDescriptions.period} />
                <Input
                  placeholder='Enter period'
                  value={params.period ?? ""}
                  onChange={(e) => setParams({ period: e.target.value })}
                />
              </div>
            )}

            {params.data_source !== "local" && (
              <div className='grid gap-2'>
                <LabelWithTooltip label='Interval' description={fieldDescriptions.interval} />
                <Input
                  placeholder='Enter interval'
                  value={params.interval ?? ""}
                  onChange={(e) => setParams({ interval: e.target.value })}
                />
              </div>
            )}
          </div>
        </>
      )}

      {/* Sampling params — only in single mode (in batch, these are per-item in BatchItemCard) */}
      {isSingle && (
        <>
          <div className='grid grid-cols-2 gap-2'>
            <div className='grid gap-2'>
              <LabelWithTooltip label='Temperature' description={fieldDescriptions.temperature} />
              <Input
                type='number'
                placeholder='Enter temperature'
                value={params.temperature ?? ""}
                onChange={(e) => setParams({ temperature: e.target.value === "" ? undefined : Number(e.target.value) })}
              />
            </div>
            <div className='grid gap-2'>
              <LabelWithTooltip label='Top P' description={fieldDescriptions.topP} />
              <Input
                type='number'
                placeholder='Enter top p'
                value={params.top_p ?? ""}
                onChange={(e) => setParams({ top_p: e.target.value === "" ? undefined : Number(e.target.value) })}
              />
            </div>
          </div>

          <div className='grid grid-cols-2 gap-2'>
            <div className='grid gap-2'>
              <LabelWithTooltip label='Top K' description={fieldDescriptions.topK} />
              <Input
                type='number'
                placeholder='Enter top k'
                value={params.top_k ?? ""}
                onChange={(e) => setParams({ top_k: e.target.value === "" ? undefined : Number(e.target.value) })}
              />
            </div>
            <div className='grid gap-2'>
              <LabelWithTooltip label='Samples' description={fieldDescriptions.sampleCount} />
              <Input
                type='number'
                placeholder='Enter samples'
                value={params.sample_count ?? ""}
                onChange={(e) =>
                  setParams({ sample_count: e.target.value === "" ? undefined : Number(e.target.value) })
                }
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default DataSelector;
