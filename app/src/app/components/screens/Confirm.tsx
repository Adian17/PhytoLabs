import { useNavigate, useLocation } from "react-router";
import { Check, RotateCcw, AlertTriangle } from "lucide-react";
import { motion } from "motion/react";
import type { ConfirmState } from "@/types/analysis";

export function Confirm() {
  const navigate = useNavigate();
  const location = useLocation();
  const state = (location.state ?? {}) as ConfirmState;

  const imageUrl =
    state.imageUrl ??
    "https://images.unsplash.com/photo-1657138691799-d3c0fc999d12?w=800&h=1200&fit=crop";

  return (
    <div className="flex flex-col h-full bg-black min-h-screen">
      <div className="absolute top-0 w-full z-10 flex justify-between items-center p-6 bg-gradient-to-b from-black/60 to-transparent text-white">
        <h2 className="text-lg font-semibold tracking-wide drop-shadow-md">Confirm Photo</h2>
      </div>

      {state.error && (
        <div className="mx-4 mt-20 mb-2 bg-red-950/90 border border-red-500/40 text-red-100 rounded-xl p-3 flex gap-2 text-sm z-20 relative">
          <AlertTriangle className="w-5 h-5 shrink-0" />
          <p>{state.error}</p>
        </div>
      )}

      <div className="flex-1 relative bg-black flex items-center justify-center p-4">
        <motion.img
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          src={imageUrl}
          alt="Captured leaf"
          className="w-full h-full max-h-[70vh] object-contain rounded-2xl"
        />
      </div>

      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="bg-white rounded-t-[32px] p-6 pb-10 flex flex-col gap-4 shadow-[0_-8px_30px_rgba(0,0,0,0.1)]"
      >
        <div className="text-center mb-2">
          <p className="text-gray-600 font-medium text-sm">
            Make sure the leaf is in focus and well-lit.
          </p>
        </div>

        <div className="flex gap-4">
          <button
            type="button"
            onClick={() => navigate("/capture")}
            className="flex-1 py-4 bg-gray-100 text-gray-800 rounded-2xl font-semibold flex items-center justify-center gap-2 hover:bg-gray-200 active:bg-gray-300 transition-colors"
          >
            <RotateCcw className="w-5 h-5" />
            Retake
          </button>
          <button
            type="button"
            onClick={() =>
              navigate("/analyzing", {
                state: { imageUrl, imageFile: state.imageFile },
              })
            }
            className="flex-1 py-4 bg-green-600 text-white rounded-2xl font-semibold flex items-center justify-center gap-2 hover:bg-green-700 shadow-lg shadow-green-600/30 active:scale-[0.98] transition-all"
          >
            <Check className="w-5 h-5" />
            Analyze
          </button>
        </div>
      </motion.div>
    </div>
  );
}
