import { useNavigate, useLocation } from "react-router";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Search, Sparkles, Activity } from "lucide-react";
import { analyzeImage, overlayDataUrl, urlToBlob } from "@/lib/api";
import type { AnalyzingState } from "@/types/analysis";

const STEPS = [
  { text: "Isolating the leaf...", icon: Search },
  { text: "Detecting lesions...", icon: Activity },
  { text: "Scoring & grading...", icon: Sparkles },
];

const MIN_ANIMATION_MS = 6500;

export function Analyzing() {
  const navigate = useNavigate();
  const location = useLocation();
  const { imageUrl, imageFile } = (location.state ?? {}) as AnalyzingState;

  const [step, setStep] = useState(0);

  useEffect(() => {
    if (!imageUrl) {
      navigate("/capture", { replace: true });
      return;
    }

    const timers = [
      setTimeout(() => setStep(1), 2000),
      setTimeout(() => setStep(2), 4000),
    ];

    let cancelled = false;

    async function run() {
      try {
        const blob = imageFile ?? (await urlToBlob(imageUrl));
        const filename = imageFile?.name ?? "leaf.jpg";
        const minDelay = new Promise<void>((r) => setTimeout(r, MIN_ANIMATION_MS));

        const [analysis] = await Promise.all([
          analyzeImage(blob, filename),
          minDelay,
        ]);

        if (cancelled) return;

        navigate("/results", {
          replace: true,
          state: {
            analysis,
            previewUrl: imageUrl,
            overlayUrl: overlayDataUrl(analysis.overlay_image_base64),
          },
        });
      } catch (err) {
        if (cancelled) return;
        navigate("/confirm", {
          replace: true,
          state: {
            imageUrl,
            imageFile,
            error:
              err instanceof Error
                ? err.message
                : "Could not reach the analysis server. Is the API running?",
          },
        });
      }
    }

    run();

    return () => {
      cancelled = true;
      timers.forEach(clearTimeout);
    };
  }, [navigate, imageUrl, imageFile]);

  const preview =
    imageUrl ??
    "https://images.unsplash.com/photo-1657138691799-d3c0fc999d12?w=800&h=1200&fit=crop";
  const CurrentIcon = STEPS[step].icon;

  return (
    <div className="flex flex-col h-full bg-gray-900 text-white relative overflow-hidden min-h-screen">
      <div className="absolute inset-0 z-0">
        <img
          src={preview}
          alt="Analyzing"
          className="w-full h-full object-cover opacity-30 grayscale blur-sm"
        />
        <div className="absolute inset-0 bg-gray-900/60" />
      </div>

      <motion.div
        animate={{ y: ["0%", "100%", "0%"] }}
        transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
        className="absolute top-0 left-0 w-full h-1/3 bg-gradient-to-b from-transparent via-green-500/20 to-transparent z-10 border-b border-green-500/50 blur-[2px]"
      />

      <div className="relative z-20 flex-1 flex flex-col items-center justify-center p-8 text-center">
        <div className="relative w-32 h-32 mb-12 flex items-center justify-center">
          <motion.div
            animate={{ scale: [1, 2], opacity: [0.8, 0] }}
            transition={{ repeat: Infinity, duration: 2, ease: "easeOut" }}
            className="absolute inset-0 border-4 border-green-500 rounded-full"
          />
          <motion.div
            animate={{ scale: [1, 2.5], opacity: [0.5, 0] }}
            transition={{ repeat: Infinity, duration: 2, ease: "easeOut", delay: 0.5 }}
            className="absolute inset-0 border-4 border-green-400 rounded-full"
          />

          <div className="relative w-20 h-20 bg-green-500 rounded-full flex items-center justify-center shadow-[0_0_40px_rgba(34,197,94,0.6)]">
            <AnimatePresence mode="wait">
              <motion.div
                key={step}
                initial={{ scale: 0.5, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.5, opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <CurrentIcon className="w-10 h-10 text-white" />
              </motion.div>
            </AnimatePresence>
          </div>
        </div>

        <div className="h-16 flex items-center justify-center">
          <AnimatePresence mode="wait">
            <motion.h2
              key={step}
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -20, opacity: 0 }}
              transition={{ duration: 0.4 }}
              className="text-2xl font-semibold tracking-wide"
            >
              {STEPS[step].text}
            </motion.h2>
          </AnimatePresence>
        </div>

        <div className="flex gap-3 mt-12">
          {STEPS.map((_, i) => (
            <motion.div
              key={i}
              className={`w-3 h-3 rounded-full ${i <= step ? "bg-green-500" : "bg-gray-700"}`}
              animate={i === step ? { scale: [1, 1.3, 1] } : {}}
              transition={{ repeat: i === step ? Infinity : 0, duration: 1 }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
