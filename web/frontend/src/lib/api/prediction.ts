import {
  PricePredictionRequestSchema,
  PricePredictionResponseSchema,
  PredictionOptionsResponseSchema,
  PricePredictionBatchRequestSchema,
  type PricePredictionResponse,
  type PredictionOptionsResponse,
  type PricePredictionRequest,
  PricePredictionBatchRequest,
  PricePredictionBatchResponse,
  PricePredictionBatchResponseSchema,
  UploadedDataFileResponseSchema,
  type UploadedDataFileResponse
} from "@/schemas/predictionSchema";

import { handleResponse, endpoint } from "@/utils/handleAPI";

export async function getPredictionOptions(): Promise<PredictionOptionsResponse> {
  const response = await fetch(endpoint("/prediction/options"), {
    method: "GET",
    cache: "no-store"
  });

  return handleResponse(response, PredictionOptionsResponseSchema);
}

export async function uploadLocalDataFile(file: File): Promise<UploadedDataFileResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(endpoint("/prediction/local-data/upload"), {
    method: "POST",
    body: formData,
    cache: "no-store"
  });

  return handleResponse(response, UploadedDataFileResponseSchema);
}

export async function predictPrice(request: PricePredictionRequest): Promise<PricePredictionResponse> {
  const body = PricePredictionRequestSchema.parse(request);
  const response = await fetch(endpoint("/prediction/price"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body),
    cache: "no-store"
  });

  return handleResponse(response, PricePredictionResponseSchema);
}

export async function predictPriceBatch(request: PricePredictionBatchRequest): Promise<PricePredictionBatchResponse> {
  const body = PricePredictionBatchRequestSchema.parse(request);
  const response = await fetch(endpoint("/prediction/price/batch"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body),
    cache: "no-store"
  });

  return handleResponse(response, PricePredictionBatchResponseSchema);
}
