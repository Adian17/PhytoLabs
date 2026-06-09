import { useEffect } from "react";
import { useNavigate, useLocation } from "react-router";
import { ArrowRight, AlertTriangle, Info, CheckCircle2, X } from "lucide-react";
import { motion } from "motion/react";
import type { AnalysisState } from "@/types/analysis";

export function Results() {
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as AnalysisState | null;

  useEffect(() => {
    if (!state?.analysis) {
      navigate("/", { replace: true });
    }
  }, [state, navigate]);

  if (!state?.analysis) {
    return null;
  }

  const { analysis, overlayUrl } = state;
  const data = analysis;

  const getVerdictConfig = (label: string) => {
    switch (label) {
      case "healthy":
        return {
          color: "text-green-600",
          bg: "bg-green-100",
          title: "Healthy",
          icon: CheckCircle2,
        };
      case "suspicious":
        return {
          color: "text-amber-600",
          bg: "bg-amber-100",
          title: "Suspicious",
          icon: Info,
        };
      case "diseased":
        return {
          color: "text-red-600",
          bg: "bg-red-100",
          title: "Rust Detected",
          icon: AlertTriangle,
        };
      default:
        return {
          color: "text-gray-600",
          bg: "bg-gray-100",
          title: "Unknown",
          icon: Info,
        };
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "none":
        return "bg-green-100 text-green-700 border-green-200";
      case "mild":
        return "bg-yellow-100 text-yellow-700 border-yellow-200";
      case "moderate":
        return "bg-orange-100 text-orange-700 border-orange-200";
      case "severe":
        return "bg-red-100 text-red-700 border-red-200";
      default:
        return "bg-gray-100 text-gray-700 border-gray-200";
    }
  };

  const verdict = getVerdictConfig(data.label);
  const VerdictIcon = verdict.icon;

  return (
    <div className="flex flex-col h-full bg-gray-50 overflow-y-auto pb-24 relative min-h-screen">
      <div className="absolute top-0 w-full z-10 flex justify-between items-center p-4">
        <button
          type="button"
          onClick={() => navigate("/")}
          className="w-10 h-10 bg-white/80 backdrop-blur-md rounded-full flex items-center justify-center shadow-sm"
        >
          <X className="w-5 h-5 text-gray-700" />
        </button>
      </div>

      <div className="w-full h-[35vh] relative rounded-b-[40px] overflow-hidden shadow-lg">
        <img src={overlayUrl} alt="Lesion overlay" className="w-full h-full object-cover" />
      </div>

      <div className="px-6 -mt-8 relative z-20">
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="bg-white rounded-[32px] p-6 shadow-xl border border-gray-100"
        >
          <div className="flex items-center gap-3 mb-2">
            <div className={`p-2 rounded-full ${verdict.bg}`}>
              <VerdictIcon className={`w-6 h-6 ${verdict.color}`} />
            </div>
            <h1 className={`text-3xl font-bold tracking-tight ${verdict.color}`}>
              {verdict.title}
            </h1>
          </div>

          <div className="mb-6 ml-11">
            <span
              className={`px-3 py-1 text-xs font-semibold uppercase tracking-wider rounded-full border ${getSeverityColor(data.severity)}`}
            >
              {data.severity} Severity
            </span>
          </div>

          <div className="mb-8">
            <div className="flex justify-between text-sm font-medium mb-2">
              <span className="text-gray-600">Model Confidence</span>
              <span className="text-gray-900">{Math.round(data.probability * 100)}%</span>
            </div>
            <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${data.probability * 100}%` }}
                transition={{ duration: 1, ease: "easeOut" }}
                className={`h-2.5 rounded-full ${data.probability > 0.8 ? "bg-green-500" : "bg-yellow-500"}`}
              />
            </div>
          </div>

          <div className="h-px bg-gray-100 w-full mb-6" />

          <div className="mb-6">
            <h3 className="text-sm font-semibold text-gray-900 mb-4 uppercase tracking-wider">
              Evidence
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-50 rounded-2xl p-4 border border-gray-100">
                <p className="text-3xl font-bold text-gray-900 mb-1">
                  {data.severity_percent.toFixed(1)}
                  <span className="text-lg text-gray-500">%</span>
                </p>
                <p className="text-xs text-gray-500 font-medium">Leaf surface affected</p>
              </div>
              <div className="bg-gray-50 rounded-2xl p-4 border border-gray-100">
                <p className="text-3xl font-bold text-gray-900 mb-1">
                  {Math.round(data.features.blob_count)}
                </p>
                <p className="text-xs text-gray-500 font-medium">Lesions detected</p>
              </div>
            </div>
          </div>

          <div className="bg-amber-50 rounded-xl p-3 flex gap-3 text-sm text-amber-800 border border-amber-100">
            <Info className="w-5 h-5 shrink-0 text-amber-600" />
            <p>{data.message}</p>
          </div>
        </motion.div>
      </div>

      <div className="fixed bottom-0 w-full max-w-md p-6 bg-gradient-to-t from-gray-50 via-gray-50 to-transparent">
        <button
          type="button"
          onClick={() => navigate("/treatment", { state })}
          className="w-full bg-gray-900 text-white rounded-2xl py-4 font-semibold text-lg flex items-center justify-center gap-2 hover:bg-black transition-colors shadow-xl active:scale-[0.98]"
        >
          View treatment plan
          <ArrowRight className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}
