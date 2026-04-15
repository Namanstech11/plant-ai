import { useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { Leaf, BarChart3, ScrollText, Github, Wifi, WifiOff } from "lucide-react";
import ImageUpload from "@/components/ImageUpload";
import PredictionResult from "@/components/PredictionResult";
import ModelDashboard from "@/components/ModelDashboard";
import PredictionLog from "@/components/PredictionLog";
import { predictDisease, getModelMetrics, checkHealth } from "@/lib/apiClient";
import type { ModelMetrics } from "@/lib/apiClient";
import { toast } from "@/hooks/use-toast";

type Tab = "detect" | "dashboard" | "history";

interface LogEntry {
  id: string;
  timestamp: string;
  imageName: string;
  prediction: string;
  confidence: number;
  model: string;
}

const Index = () => {
  const [activeTab, setActiveTab] = useState<Tab>("detect");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null);
  const [result, setResult] = useState<{
    name: string;
    confidence: number;
    severity: "low" | "medium" | "high";
    treatment: string;
    description: string;
  } | null>(null);
  const [allPredictions, setAllPredictions] = useState<
    { model: string; disease: string; confidence: number }[]
  >([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [modelMetrics, setModelMetrics] = useState<
    { name: string; accuracy: number; precision: number; recall: number; f1Score: number }[]
  >([]);

  const fetchMetrics = useCallback(() => {
    getModelMetrics()
      .then((data) => {
        setModelMetrics(
          data.models.map((m: ModelMetrics) => ({
            name: m.name,
            accuracy: m.accuracy,
            precision: m.precision,
            recall: m.recall,
            f1Score: m.f1_score,
          }))
        );
      })
      .catch(() => {
        setModelMetrics([
          { name: "CNN (MobileNetV2)", accuracy: 0, precision: 0, recall: 0, f1Score: 0 },
          { name: "Random Forest", accuracy: 0, precision: 0, recall: 0, f1Score: 0 },
          { name: "SVM (RBF)", accuracy: 0, precision: 0, recall: 0, f1Score: 0 },
          { name: "XGBoost", accuracy: 0, precision: 0, recall: 0, f1Score: 0 },
          { name: "ViT", accuracy: 0, precision: 0, recall: 0, f1Score: 0 },
          { name: "CNN-ViT Hybrid", accuracy: 0, precision: 0, recall: 0, f1Score: 0 },
        ]);
      });
  }, []);

  // Check backend health on mount and refresh metrics periodically
  useEffect(() => {
    checkHealth().then(setBackendConnected);
    fetchMetrics();

    const interval = setInterval(() => {
      checkHealth().then(setBackendConnected);
      fetchMetrics();
    }, 30000); // Refresh every 30s

    return () => clearInterval(interval);
  }, [fetchMetrics]);

  const handleImageAnalysis = useCallback(
    async (file: File) => {
      if (!backendConnected) {
        toast({
          title: "Backend not connected",
          description: "Start the ML backend: cd ml-backend && uvicorn api:app --port 8000",
          variant: "destructive",
        });
        return;
      }

      setIsAnalyzing(true);
      setResult(null);

      try {
        const prediction = await predictDisease(file);

        const isUncertain = prediction.raw_class === "unknown";
        const isHealthy = prediction.raw_class.toLowerCase().includes("healthy");

        setResult({
          name: prediction.predicted_disease,
          confidence: prediction.confidence,
          severity: prediction.severity,
          treatment: prediction.treatment,
          description: isUncertain
            ? `The models were not confident enough to confirm a disease. Highest model confidence was ${(prediction.confidence * 100).toFixed(1)}%.`
            : isHealthy
              ? `Healthy leaf detected by ${prediction.best_model} with ${(prediction.confidence * 100).toFixed(1)}% confidence.`
              : `Detected by ${prediction.best_model} with ${(prediction.confidence * 100).toFixed(1)}% confidence using real ML inference.`,
        });

        setAllPredictions(
          Object.entries(prediction.model_predictions).map(([model, pred]) => ({
            model,
            disease: pred.disease,
            confidence: pred.confidence,
          }))
        );

        const newLog: LogEntry = {
          id: Date.now().toString(),
          timestamp: new Date().toLocaleTimeString(),
          imageName: file.name,
          prediction: prediction.predicted_disease,
          confidence: prediction.confidence,
          model: prediction.best_model,
        };
        setLogs((prev) => [newLog, ...prev]);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Analysis failed";
        toast({ title: "Prediction Error", description: message, variant: "destructive" });
      } finally {
        setIsAnalyzing(false);
      }
    },
    [backendConnected]
  );

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: "detect", label: "Detect Disease", icon: <Leaf className="w-4 h-4" /> },
    { id: "dashboard", label: "Model Dashboard", icon: <BarChart3 className="w-4 h-4" /> },
    { id: "history", label: "History", icon: <ScrollText className="w-4 h-4" /> },
  ];

  return (
    <div className="min-h-screen gradient-hero">
      <header className="border-b border-border bg-card/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container max-w-5xl mx-auto flex items-center justify-between h-14 px-4">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg gradient-leaf">
              <Leaf className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="font-display font-bold text-foreground text-lg">
              PlantGuard AI
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs">
              {backendConnected === null ? (
                <span className="text-muted-foreground">Checking API...</span>
              ) : backendConnected ? (
                <>
                  <Wifi className="w-3.5 h-3.5 text-success" />
                  <span className="text-success font-medium">ML Backend Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-3.5 h-3.5 text-destructive" />
                  <span className="text-destructive font-medium">Backend Offline</span>
                </>
              )}
            </div>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <Github className="w-5 h-5" />
            </a>
          </div>
        </div>
      </header>

      <main className="container max-w-5xl mx-auto px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-10"
        >
          <h1 className="text-4xl md:text-5xl font-display font-bold text-foreground mb-3">
            Plant Disease Detection
          </h1>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Upload a leaf image for real ML inference using CNN, ViT, Hybrid CNN-ViT,
            Random Forest, SVM, and XGBoost — 6 models trained on the PlantVillage dataset.
          </p>
        </motion.div>

        <div className="flex justify-center mb-8">
          <div className="inline-flex gap-1 p-1 rounded-xl bg-secondary border border-border">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  activeTab === tab.id
                    ? "bg-card text-foreground shadow-soft"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {activeTab === "detect" && (
          <div className="space-y-8">
            <ImageUpload onImageSelect={handleImageAnalysis} isAnalyzing={isAnalyzing} />
            {result && (
              <PredictionResult result={result} allPredictions={allPredictions} />
            )}
            {!result && !isAnalyzing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="text-center py-8"
              >
                {!backendConnected && (
                  <div className="mb-6 p-4 rounded-xl bg-destructive/10 border border-destructive/20 max-w-lg mx-auto">
                    <p className="text-sm text-destructive font-medium mb-1">ML Backend Required</p>
                    <p className="text-xs text-muted-foreground">
                      Run the backend: <code className="bg-secondary px-1.5 py-0.5 rounded text-foreground">cd ml-backend && python train_models.py && uvicorn api:app --port 8000</code>
                    </p>
                  </div>
                )}
                {modelMetrics.length > 0 && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-2xl mx-auto">
                    {modelMetrics.map((m) => (
                      <div key={m.name} className="p-4 rounded-xl bg-card border border-border shadow-soft text-center">
                        <p className="text-2xl font-display font-bold text-primary">
                          {m.accuracy > 0 ? `${(m.accuracy * 100).toFixed(0)}%` : "—"}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">{m.name}</p>
                      </div>
                    ))}
                  </div>
                )}
              </motion.div>
            )}
          </div>
        )}

        {activeTab === "dashboard" && <ModelDashboard models={modelMetrics} />}
        {activeTab === "history" && <PredictionLog logs={logs} />}
      </main>

      <footer className="border-t border-border py-6 mt-12">
        <div className="container max-w-5xl mx-auto px-4 text-center text-xs text-muted-foreground">
          PlantGuard AI · Real ML inference with CNN, ViT, Hybrid, RF, SVM & XGBoost · PlantVillage Dataset (38 classes)
        </div>
      </footer>
    </div>
  );
};

export default Index;
