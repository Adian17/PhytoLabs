import { useRef } from "react";
import { useNavigate } from "react-router";
import { Camera, Image as ImageIcon, Leaf, ArrowRight } from "lucide-react";
import { motion } from "motion/react";

export function Home() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  function openGallery(file: File) {
    const url = URL.createObjectURL(file);
    navigate("/confirm", { state: { imageUrl: url, imageFile: file } });
  }

  return (
    <div className="flex flex-col h-full bg-green-50 min-h-screen">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) openGallery(file);
          e.target.value = "";
        }}
      />

      <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", damping: 20, stiffness: 100 }}
          className="w-24 h-24 bg-green-600 rounded-3xl flex items-center justify-center mb-6 shadow-lg shadow-green-600/30"
        >
          <Leaf className="w-12 h-12 text-white" />
        </motion.div>

        <motion.h1
          initial={{ y: 10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-3xl font-extrabold text-gray-900 tracking-tight mb-3"
        >
          PhytoLabs
        </motion.h1>

        <motion.p
          initial={{ y: 10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-gray-600 text-lg mb-10 max-w-[260px]"
        >
          Scan a wheat leaf, get an instant rust diagnosis.
        </motion.p>
      </div>

      <motion.div
        initial={{ y: 40, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="bg-white p-6 rounded-t-[32px] shadow-[0_-8px_30px_rgba(0,0,0,0.04)] pb-8"
      >
        <div className="space-y-4">
          <button
            type="button"
            onClick={() => navigate("/capture")}
            className="w-full bg-green-600 text-white rounded-2xl py-4 font-semibold text-lg flex items-center justify-center gap-2 hover:bg-green-700 transition-colors shadow-md shadow-green-600/20 active:scale-[0.98]"
          >
            <Camera className="w-5 h-5" />
            Scan a leaf
          </button>

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="w-full bg-green-50 text-green-700 rounded-2xl py-4 font-semibold text-lg flex items-center justify-center gap-2 hover:bg-green-100 transition-colors active:scale-[0.98]"
          >
            <ImageIcon className="w-5 h-5" />
            Upload from gallery
          </button>
        </div>

        <div className="mt-8">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 px-2">
            Recent Scans
          </h3>
          <p className="text-sm text-gray-400 px-2">
            Scan history coming soon — run an analysis to see live results.
          </p>
          <div className="mt-4 flex items-center gap-2 text-gray-300 text-sm px-2">
            <ArrowRight className="w-4 h-4" />
            <span>Results powered by your PhytoLabs API</span>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
