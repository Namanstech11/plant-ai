import { useState, useCallback } from "react";
import { Upload, X, Loader2, Leaf } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  isAnalyzing: boolean;
}

const ImageUpload = ({ onImageSelect, isAnalyzing }: ImageUploadProps) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      const url = URL.createObjectURL(file);
      setPreview(url);
      onImageSelect(file);
    },
    [onImageSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const clearImage = () => {
    setPreview(null);
  };

  return (
    <div className="w-full max-w-lg mx-auto">
      <AnimatePresence mode="wait">
        {!preview ? (
          <motion.label
            key="upload"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            htmlFor="leaf-upload"
            className={`relative flex flex-col items-center justify-center w-full h-72 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-300 ${
              isDragging
                ? "border-primary bg-primary/5 scale-[1.02]"
                : "border-border hover:border-primary/50 hover:bg-secondary/50"
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center gap-3">
              <div className="p-4 rounded-full bg-secondary">
                <Leaf className="w-8 h-8 text-primary" />
              </div>
              <div className="text-center">
                <p className="text-base font-medium text-foreground">
                  Drop a leaf image here
                </p>
                <p className="text-sm text-muted-foreground mt-1">
                  or click to browse · JPG, PNG up to 10MB
                </p>
              </div>
              <Button variant="outline" size="sm" type="button" className="mt-2">
                <Upload className="w-4 h-4 mr-1" />
                Choose File
              </Button>
            </div>
            <input
              id="leaf-upload"
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleChange}
            />
          </motion.label>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="relative rounded-xl overflow-hidden shadow-card"
          >
            <img
              src={preview}
              alt="Uploaded leaf"
              className="w-full h-72 object-cover"
            />
            {isAnalyzing && (
              <div className="absolute inset-0 bg-foreground/40 backdrop-blur-sm flex items-center justify-center">
                <div className="flex flex-col items-center gap-3">
                  <Loader2 className="w-8 h-8 text-primary-foreground animate-spin" />
                  <p className="text-primary-foreground font-medium text-sm">
                    Analyzing leaf...
                  </p>
                </div>
              </div>
            )}
            {!isAnalyzing && (
              <button
                onClick={clearImage}
                className="absolute top-3 right-3 p-1.5 rounded-full bg-foreground/60 text-background hover:bg-foreground/80 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ImageUpload;
