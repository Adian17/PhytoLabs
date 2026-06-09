import type { AnalyzeResponse } from "@/types/analysis";

/** Base URL for the PhytoLabs FastAPI backend. */
const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

export function overlayDataUrl(base64: string): string {
  return `data:image/png;base64,${base64}`;
}

export async function urlToBlob(url: string): Promise<Blob> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error("Could not load the selected image.");
  }
  return res.blob();
}

export async function analyzeImage(
  imageBlob: Blob,
  filename = "leaf.jpg",
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("image", imageBlob, filename);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let detail = `Analysis failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }

  return res.json() as Promise<AnalyzeResponse>;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) return false;
    const body = await res.json();
    return body.models_loaded === true;
  } catch {
    return false;
  }
}
