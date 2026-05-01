"use client";
import { Field, FieldLabel, FieldTitle } from "@/components/ui/field";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { usePricePredictionStore } from "@/stores/pricePredictionStore";
import { ModelNameEnum } from "@/schemas/predictionSchema";
import z from "zod";

const modelItems = [
  {
    value: "kronos-mini" as const,
    title: "Kronos-mini",
    description: "Fast",
    params: "4.1M",
    context: "2048"
  },
  {
    value: "kronos-small" as const,
    title: "Kronos-Small",
    description: "Balanced",
    params: "27.7M",
    context: "512"
  },
  {
    value: "kronos-base" as const,
    title: "Kronos-Base",
    description: "Quality",
    params: "102.3M",
    context: "512"
  }
];

export default function ModelSelector() {
  const mode = usePricePredictionStore((state) => state.mode);
  const singleModelName = usePricePredictionStore((state) => state.params.model_name);
  const batchModelName = usePricePredictionStore((state) => state.batchParams.model_name);
  const setParam = usePricePredictionStore((state) => state.setParam);
  const setBatchParam = usePricePredictionStore((state) => state.setBatchParam);

  const model_name = mode === "single" ? singleModelName : batchModelName;
  const handleChange = (value: string) => {
    const typed = value as z.infer<typeof ModelNameEnum>;
    if (mode === "single") {
      setParam("model_name", typed);
    } else {
      setBatchParam("model_name", typed);
    }
  };

  return (
    <RadioGroup
      value={model_name}
      onValueChange={handleChange}
      className='grid w-full grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-2 lg:gap-4'
    >
      {modelItems.map((item) => (
        <FieldLabel key={item.value} htmlFor={item.value} className='relative py-1 cursor-pointer overflow-hidden'>
          <Field orientation='horizontal'>
            <FieldTitle className='flex flex-col items-start'>
              <div className='w-full flex items-center gap-2'>
                <RadioGroupItem value={item.value} id={item.value} />
                <span className='text-sm font-semibold'>{item.title}</span>
              </div>

              <div className='flex justify-between gap-2 w-full'>
                <div className='flex flex-col gap-0'>
                  <span className='text-muted-foreground text-xs'>Params</span>
                  <span className='text-sm text-center font-semibold'>{item.params}</span>
                </div>
                <div className='flex flex-col gap-0'>
                  <span className='text-muted-foreground text-xs'>Context</span>
                  <span className='text-sm text-center font-semibold'>{item.context}</span>
                </div>
              </div>
            </FieldTitle>
          </Field>
        </FieldLabel>
      ))}
    </RadioGroup>
  );
}
