import { z } from "zod";

import { PricePredictionResponseSchema, PricePredictionBatchResponseSchema } from "./predictionSchema";

export const ResultTypeSchema = z.enum(["predict/price", "predict/price/batch"]);
export type ResultType = z.infer<typeof ResultTypeSchema>;

export const SavedResultMetaSchema = z.object({
  id: z.string(),
  type: ResultTypeSchema,
  label: z.string().nullable(),
  created_at: z.string()
});

export const SavedResultDetailSchema = z.object({
  id: z.string(),
  type: ResultTypeSchema,
  label: z.string().nullable(),
  created_at: z.string(),
  data: z.union([PricePredictionResponseSchema, PricePredictionBatchResponseSchema])
});


export const SaveResultRequestSchema = z.object({
  type: ResultTypeSchema,
  label: z.string().nullable().optional(),
  data: z.union([PricePredictionResponseSchema, PricePredictionBatchResponseSchema])
});

export type SavedResultMeta = z.infer<typeof SavedResultMetaSchema>;
export type SavedResultDetail = z.infer<typeof SavedResultDetailSchema>;
export type SaveResultRequest = z.infer<typeof SaveResultRequestSchema>;
