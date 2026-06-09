import { useRef } from "react";
import { useNavigate } from "react-router";
import { Camera, X, ImageIcon, Flashlight } from "lucide-react";
import { useState } from "react";

export function Capture() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [flash, setFlash] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewFile, setPreviewFile] = useState<File | null>(null);

  function handleFile(file: File) {
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setPreviewFile(file);
    navigate("/confirm", { state: { imageUrl: url, imageFile: file } });
  }

  const cameraFeedUrl =
    previewUrl ??
    "https://images.unsplash.com/photo-1657138691799-d3c0fc999d12?w=800&h=1200&fit=crop";

  return (
    <div className="flex flex-col h-full bg-black text-white relative min-h-screen">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
          e.target.value = "";
        }}
      />

      <div className="absolute top-0 w-full z-10 flex justify-between items-center p-6 bg-gradient-to-b from-black/60 to-transparent">
        <button
          type="button"
          onClick={() => navigate(-1)}
          className="w-10 h-10 bg-black/40 backdrop-blur-md rounded-full flex items-center justify-center hover:bg-black/60 transition-colors"
        >
          <X className="w-5 h-5 text-white" />
        </button>
        <button
          type="button"
          onClick={() => setFlash(!flash)}
          className={`w-10 h-10 rounded-full flex items-center justify-center transition-colors ${
            flash ? "bg-yellow-500 text-black" : "bg-black/40 backdrop-blur-md text-white hover:bg-black/60"
          }`}
        >
          <Flashlight className="w-5 h-5" />
        </button>
      </div>

      <div className="flex-1 relative overflow-hidden">
        <img
          src={cameraFeedUrl}
          alt="Camera preview"
          className="absolute inset-0 w-full h-full object-cover opacity-80"
        />

        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none p-8">
          <div className="w-full max-w-[240px] aspect-[1/2.5] border-2 border-white/50 rounded-[40px] relative">
            <div className="absolute top-0 left-0 w-8 h-8 border-t-4 border-l-4 border-green-400 rounded-tl-[38px]" />
            <div className="absolute top-0 right-0 w-8 h-8 border-t-4 border-r-4 border-green-400 rounded-tr-[38px]" />
            <div className="absolute bottom-0 left-0 w-8 h-8 border-b-4 border-l-4 border-green-400 rounded-bl-[38px]" />
            <div className="absolute bottom-0 right-0 w-8 h-8 border-b-4 border-r-4 border-green-400 rounded-br-[38px]" />
          </div>

          <div className="bg-black/60 backdrop-blur-md text-white text-sm font-medium px-4 py-2 rounded-full mt-8 max-w-[280px] text-center shadow-lg border border-white/10">
            Fill the frame with one leaf • Plain background • Good light
          </div>
        </div>
      </div>

      <div className="bg-black pb-10 pt-6 px-8 flex justify-between items-center h-[140px]">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="w-12 h-12 bg-white/10 rounded-full flex items-center justify-center hover:bg-white/20 transition-colors"
        >
          <ImageIcon className="w-5 h-5" />
        </button>

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="w-20 h-20 rounded-full border-4 border-white flex items-center justify-center p-1 active:scale-95 transition-transform"
          aria-label="Capture leaf photo"
        >
          <div className="w-full h-full bg-white rounded-full flex items-center justify-center">
            <Camera className="w-8 h-8 text-black" />
          </div>
        </button>

        <div className="w-12 h-12" />
      </div>
    </div>
  );
}
