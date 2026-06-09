export interface AnalyzeFeatures {
  lesion_area_fraction: number;
  blob_count: number;
  blob_size_mean: number;
  blob_size_std: number;
  blob_size_max: number;
  blob_density: number;
}

export type VerdictLabel = "healthy" | "suspicious" | "diseased";
export type SeverityGrade = "none" | "mild" | "moderate" | "severe";

export interface AnalyzeResponse {
  label: VerdictLabel;
  probability: number;
  severity: SeverityGrade;
  severity_percent: number;
  features: AnalyzeFeatures;
  overlay_image_base64: string;
  message: string;
}

/** Passed through React Router location.state after a successful scan. */
export interface AnalysisState {
  analysis: AnalyzeResponse;
  previewUrl: string;
  overlayUrl: string;
}

export interface ConfirmState {
  imageUrl: string;
  imageFile?: File;
  error?: string;
}

export interface AnalyzingState {
  imageUrl: string;
  imageFile?: File;
}
