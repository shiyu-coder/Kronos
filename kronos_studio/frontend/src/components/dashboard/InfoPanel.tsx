import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { usePricePredictionStore } from "@/stores/pricePredictionStore";

const getBatchMode = (modelName: string) => {
  switch (modelName) {
    case "kronos-mini":
      return "Fast";
    case "kronos-small":
      return "Balanced";
    case "kronos-base":
      return "Quality";
    default:
      return "Unknown";
  }
};

const InfoPanel = () => {
  const singleParams = usePricePredictionStore((state) => state.params);
  const batchParamsState = usePricePredictionStore((state) => state.batchParams);
  const mode = usePricePredictionStore((state) => state.mode);
  const batchItems = usePricePredictionStore((state) => state.batchItems);
  const batchResults = usePricePredictionStore((state) => state.batchResults);
  const activeResultIndex = usePricePredictionStore((state) => state.activeResultIndex);

  const isBatch = mode === "batch";
  const params = isBatch ? batchParamsState : singleParams;

  // In batch mode, show active result's symbol or the configured batch item symbol
  const activeSymbol = isBatch
    ? batchResults && batchResults.length > 0
      ? batchResults[activeResultIndex]?.request.symbol || `Symbol #${activeResultIndex + 1}`
      : batchItems[0]?.symbol || "Not selected"
    : params.symbol || "Not selected";

  const activeDataSource = isBatch
    ? batchResults && batchResults.length > 0
      ? batchResults[activeResultIndex]?.request.data_source
      : batchItems[0]?.data_source || "binance"
    : params.data_source;

  const activeInterval = isBatch
    ? batchResults && batchResults.length > 0
      ? batchResults[activeResultIndex]?.request.interval
      : batchItems[0]?.interval || "1h"
    : params.interval;

  const DashboardData = [
    {
      label: "Selected market",
      title: activeSymbol,
      subtitle: `${activeDataSource}, ${activeInterval} candles`,
      textColor: "text-green-300"
    },
    {
      label: "Forecast window",
      title: `${params.pred_len} steps`,
      subtitle: `${params.lookback} candle context`,
      textColor: "text-cyan-300"
    },
    {
      label: "Runtime",
      title: params.device,
      subtitle: params.model_name,
      textColor: "text-amber-300"
    }
  ];

  return (
    <div className='grid w-full grid-cols-1 md:grid-cols-3 gap-2 lg:gap-4'>
      {DashboardData.map((item, index) => (
        <Card key={item.label} className='flex-1 gap-1 p-4 relative'>
          {index === 2 && (
            <div className='absolute top-2 right-2 px-2 py-0.5 bg-emerald-500/20 text-emerald-400 text-xs font-medium'>
              {getBatchMode(params.model_name)}
            </div>
          )}

          {/* Batch count badge on the first card */}
          {index === 0 && isBatch && (
            <div className='absolute top-2 right-2 px-2 py-0.5 bg-blue-500/20 text-blue-400 text-xs font-medium rounded'>
              Batch ({batchItems.length})
            </div>
          )}

          <CardHeader className='p-0'>
            <CardDescription className='text-neutral-500 text-xs capitalize'>{item.label}</CardDescription>
          </CardHeader>

          <CardContent className='p-0'>
            <CardTitle className={`text-xl font-bold text-whit uppercase`}>{item.title}</CardTitle>
            <p className={`text-sm capitalize ${item.textColor}`}>{item.subtitle}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

export default InfoPanel;
