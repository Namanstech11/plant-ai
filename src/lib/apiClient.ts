/**
 * PlantGuard AI - API Client
 * Connects to the FastAPI backend for real ML inference.
 */

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface ModelPrediction {
  disease: string;
  confidence: number;
}

export interface PredictionResult {
  predicted_disease: string;
  raw_class: string;
  confidence: number;
  severity: "low" | "medium" | "high";
  treatment: string;
  best_model: string;
  model_predictions: Record<string, ModelPrediction>;
}

export interface ModelMetrics {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
}

export interface MetricsResponse {
  models: ModelMetrics[];
  best_model: string;
}

export async function predictDisease(imageFile: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Prediction failed" }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

export async function getModelMetrics(): Promise<MetricsResponse> {
  const response = await fetch(`${API_URL}/models/metrics`);
  if (!response.ok) {
    throw new Error("Failed to fetch model metrics");
  }
  return response.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`);
    const data = await response.json();
    return data.models_loaded === true;
  } catch {
    return false;
  }
}
