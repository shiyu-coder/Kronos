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
  const params = usePricePredictionStore((state) => state.params);

  const DashboardData = [
    {
      label: "Selected market",
      title: params.symbol || "Not selected",
      subtitle: `${params.data_source}, ${params.interval} candles`,
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
    <div className='flex gap-4'>
      {DashboardData.map((item, index) => (
        <Card key={item.label} className='flex-1 gap-1 p-4 relative'>
          {index === 2 && (
            <div className='absolute top-2 right-2 px-2 py-0.5 bg-emerald-500/20 text-emerald-400 text-xs font-medium'>
              {getBatchMode(params.model_name)}
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
