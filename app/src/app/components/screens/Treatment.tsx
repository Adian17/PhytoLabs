import { useNavigate, useLocation } from "react-router";
import {
  ArrowLeft,
  Beaker,
  CalendarClock,
  ShieldAlert,
  Sparkles,
  ChevronRight,
} from "lucide-react";
import { useState, useEffect } from "react";
import { motion } from "motion/react";
import type { AnalysisState } from "@/types/analysis";

/** Placeholder treatment copy keyed by severity until LLM + RAG is wired. */
function treatmentCopy(severity: string, label: string) {
  if (label === "healthy" && severity === "none") {
    return {
      summary:
        "No significant rust detected. Continue routine monitoring during the growing season.",
      actions: [
        {
          title: "Monitor weekly",
          body: "Scout upper leaves for new orange-brown pustules, especially after warm, humid weather.",
        },
      ],
      when: "Re-scan if you notice new spots or before fungicide decision windows.",
    };
  }
  if (severity === "mild") {
    return {
      summary:
        "Trace to mild leaf rust detected. Early intervention can prevent canopy spread.",
      actions: [
        {
          title: "Scout adjacent plants",
          body: "Check neighboring leaves within 48 hours — rust spreads quickly on wind.",
        },
        {
          title: "Consider protective fungicide",
          body: "A triazole or mixed-mode product may be appropriate if weather favors rust.",
        },
      ],
      when: "Act within 3–5 days if conditions stay humid; re-scan to track severity.",
    };
  }
  return {
    summary: `Moderate to severe leaf rust (${severity}). Immediate management is recommended to protect yield.`,
    actions: [
      {
        title: "Apply systemic fungicide",
        body: "Use a product with curative and preventative activity (e.g., triazole + strobilurin mix).",
      },
      {
        title: "Monitor adjoining fields",
        body: "Rust spreads via wind — inspect neighboring paddocks within 48 hours.",
      },
    ],
    when: "Optimal window: within 2–3 days. Avoid spraying if rain is expected within 2 hours.",
  };
}

export function Treatment() {
  const navigate = useNavigate();
  const location = useLocation();
  const analysisState = location.state as AnalysisState | null;
  const analysis = analysisState?.analysis;

  const [isGenerating, setIsGenerating] = useState(true);

  useEffect(() => {
    if (!analysis) {
      navigate("/", { replace: true });
      return;
    }
    const timer = setTimeout(() => setIsGenerating(false), 2500);
    return () => clearTimeout(timer);
  }, [analysis, navigate]);

  if (!analysis) return null;

  const copy = treatmentCopy(analysis.severity, analysis.label);

  return (
    <div className="flex flex-col h-full bg-white relative min-h-screen">
      <div className="flex items-center p-4 border-b border-gray-100 sticky top-0 bg-white z-10">
        <button
          type="button"
          onClick={() => navigate(-1)}
          className="p-2 -ml-2 rounded-full hover:bg-gray-100 transition-colors"
        >
          <ArrowLeft className="w-6 h-6 text-gray-700" />
        </button>
        <h1 className="ml-2 text-lg font-bold text-gray-900">Treatment Plan</h1>
      </div>

      {isGenerating ? (
        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center bg-gray-50">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ repeat: Infinity, duration: 4, ease: "linear" }}
            className="w-16 h-16 border-4 border-blue-100 border-t-blue-600 rounded-full mb-6"
          />
          <h2 className="text-xl font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-blue-600" />
            Synthesizing plan...
          </h2>
          <p className="text-gray-500 text-sm max-w-[250px]">
            Using {analysis.severity} severity · {Math.round(analysis.probability * 100)}%
            confidence (LLM + RAG coming soon).
          </p>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto pb-8">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-6 space-y-8"
          >
            <section>
              <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-3">
                Summary
              </h2>
              <p className="text-gray-800 leading-relaxed text-lg font-medium">{copy.summary}</p>
            </section>

            <section>
              <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4">
                Recommended Actions
              </h2>
              <div className="space-y-4">
                {copy.actions.map((action) => (
                  <div key={action.title} className="flex gap-4 items-start">
                    <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center shrink-0 mt-1">
                      <ShieldAlert className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 text-base">{action.title}</h3>
                      <p className="text-gray-600 text-sm mt-1">{action.body}</p>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section>
              <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4">
                Common Products
              </h2>
              <div className="bg-gray-50 border border-gray-100 rounded-2xl overflow-hidden">
                {[
                  { name: "Prosaro® PRO", class: "Triazole", rate: "10-12 fl oz/A" },
                  { name: "Trivapro®", class: "SDHI + Triazole + Strobilurin", rate: "13.7 fl oz/A" },
                ].map((product, i) => (
                  <div
                    key={product.name}
                    className={`p-4 flex items-center justify-between ${i > 0 ? "border-t border-gray-200" : ""}`}
                  >
                    <div>
                      <h4 className="font-bold text-gray-900 flex items-center gap-2">
                        <Beaker className="w-4 h-4 text-gray-400" />
                        {product.name}
                      </h4>
                      <p className="text-xs text-gray-500 mt-1">
                        {product.class} • {product.rate}
                      </p>
                    </div>
                    <ChevronRight className="w-5 h-5 text-gray-300" />
                  </div>
                ))}
              </div>
            </section>

            <section className="bg-blue-50 border border-blue-100 rounded-2xl p-5">
              <h2 className="text-sm font-bold text-blue-800 uppercase tracking-wider mb-2 flex items-center gap-2">
                <CalendarClock className="w-4 h-4" />
                When to Act
              </h2>
              <p className="text-blue-900 text-sm font-medium">{copy.when}</p>
            </section>

            <div className="mt-8 pt-6 border-t border-gray-100">
              <p className="text-xs text-gray-400 text-center leading-relaxed">
                <strong className="text-gray-500">Disclaimer:</strong> Placeholder guidance based on
                severity grade — not yet LLM-generated. Always consult a certified agronomist.
              </p>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
