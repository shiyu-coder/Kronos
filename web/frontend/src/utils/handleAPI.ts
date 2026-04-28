const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

export class PredictionApiError extends Error {
  readonly status: number;
  readonly code?: string;
  readonly details?: unknown;

  constructor(message: string, status: number, code?: string, details?: unknown) {
    super(message);
    this.name = "PredictionApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export async function handleResponse<T>(response: Response, schema: { parse: (data: unknown) => T }): Promise<T> {
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const error = data && data.error ? data.error : null;
    throw new PredictionApiError(
      error?.message ?? `Prediction API request failed with status ${response.status}.`,
      response.status,
      error?.code,
      error?.details
    );
  }

  return schema.parse(data);
}

export function endpoint(path: string): string {
  return `${API_BASE_URL.replace(/\/$/, "")}${path}`;
}
