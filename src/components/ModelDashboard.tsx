import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend,
} from "recharts";
import { Brain, Cpu, TreePine, Zap, Eye, Layers } from "lucide-react";

interface ModelMetrics {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
}

interface ModelDashboardProps {
  models: ModelMetrics[];
}

const MODEL_DETAILS: Record<string, { icon: React.ReactNode; type: string; description: string; architecture: string[] }> = {
  "CNN (MobileNetV2)": {
    icon: <Cpu className="w-5 h-5" />,
    type: "Deep Learning",
    description: "Transfer learning with MobileNetV2 backbone, fine-tuned on PlantVillage. Extracts spatial features through depthwise separable convolutions.",
    architecture: ["MobileNetV2 Backbone (ImageNet)", "Global Average Pooling", "Dense 256 → Dense 128", "Softmax Classification"],
  },
  "Random Forest": {
    icon: <TreePine className="w-5 h-5" />,
    type: "Ensemble ML",
    description: "200-tree ensemble trained on 128-d CNN feature vectors. Uses bagging and random feature subsets for robust classification.",
    architecture: ["CNN Feature Extraction (128-d)", "200 Decision Trees", "Max Depth 30", "Majority Voting"],
  },
  "SVM (RBF)": {
    icon: <Zap className="w-5 h-5" />,
    type: "Kernel ML",
    description: "Support Vector Machine with RBF kernel operating on CNN-extracted features. Projects data into high-dimensional space for optimal separation.",
    architecture: ["CNN Feature Extraction (128-d)", "RBF Kernel (γ=scale)", "C=10 Regularization", "Probability Calibration"],
  },
  "XGBoost": {
    icon: <Layers className="w-5 h-5" />,
    type: "Gradient Boosting",
    description: "Gradient-boosted trees trained on CNN features. Sequentially corrects errors with 200 weak learners for high accuracy.",
    architecture: ["CNN Feature Extraction (128-d)", "200 Boosted Trees", "Max Depth 8, LR 0.1", "Softprob Objective"],
  },
  "ViT": {
    icon: <Eye className="w-5 h-5" />,
    type: "Transformer",
    description: "Vision Transformer splitting 128×128 images into 16×16 patches. Self-attention captures global relationships between all image regions.",
    architecture: ["Patch Embedding (16×16 → 64 tokens)", "Positional Encoding", "6 Transformer Blocks (8 heads)", "GAP → Dense 128 → Softmax"],
  },
  "CNN-ViT Hybrid": {
    icon: <Brain className="w-5 h-5" />,
    type: "Hybrid",
    description: "MobileNetV2 extracts spatial feature maps, reshaped into tokens for Transformer refinement. Combines CNN locality with Transformer global attention.",
    architecture: ["MobileNetV2 → Feature Maps (4×4×1280)", "Reshape to 16 tokens → Project 128-d", "3 Transformer Blocks (4 heads)", "GAP → Dense 128 → Softmax"],
  },
};

const colors = [
  "hsl(152, 55%, 35%)", "hsl(35, 90%, 55%)", "hsl(210, 70%, 50%)",
  "hsl(280, 60%, 50%)", "hsl(0, 72%, 51%)", "hsl(180, 55%, 40%)",
];

const ModelDashboard = ({ models }: ModelDashboardProps) => {
  const barData = models.map((m) => ({
    name: m.name,
    Accuracy: +(m.accuracy * 100).toFixed(1),
    F1: +(m.f1Score * 100).toFixed(1),
  }));

  const radarData = [
    { metric: "Accuracy", ...Object.fromEntries(models.map((m) => [m.name, +(m.accuracy * 100).toFixed(1)])) },
    { metric: "Precision", ...Object.fromEntries(models.map((m) => [m.name, +(m.precision * 100).toFixed(1)])) },
    { metric: "Recall", ...Object.fromEntries(models.map((m) => [m.name, +(m.recall * 100).toFixed(1)])) },
    { metric: "F1 Score", ...Object.fromEntries(models.map((m) => [m.name, +(m.f1Score * 100).toFixed(1)])) },
  ];

  const bestModel = models.reduce((best, m) => m.f1Score > best.f1Score ? m : best, models[0]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="w-full max-w-4xl mx-auto space-y-8"
    >
      <div>
        <h2 className="text-2xl font-display font-bold text-foreground mb-2">
          Model Performance Comparison
        </h2>
        <p className="text-sm text-muted-foreground">
          6 models trained on PlantVillage dataset · Real-time metrics from ML backend
        </p>
      </div>

      {/* Best model banner */}
      {bestModel && bestModel.f1Score > 0 && (
        <Card className="p-4 border-primary/30 bg-primary/5">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10">
              {MODEL_DETAILS[bestModel.name]?.icon || <Brain className="w-5 h-5" />}
            </div>
            <div>
              <p className="text-xs font-medium text-primary uppercase tracking-wide">Best Model</p>
              <p className="text-sm font-bold text-foreground">
                {bestModel.name} — F1: {(bestModel.f1Score * 100).toFixed(1)}% · Accuracy: {(bestModel.accuracy * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="p-5 shadow-card border-border">
          <h3 className="text-sm font-semibold text-foreground uppercase tracking-wide mb-4">
            Accuracy & F1 Score
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={barData} barGap={4}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(145, 15%, 88%)" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: "hsl(150, 10%, 45%)" }} angle={-20} textAnchor="end" height={60} />
              <YAxis tick={{ fontSize: 11, fill: "hsl(150, 10%, 45%)" }} domain={[0, 100]} />
              <Tooltip contentStyle={{ backgroundColor: "hsl(0, 0%, 100%)", border: "1px solid hsl(145, 15%, 88%)", borderRadius: "8px", fontSize: "12px" }} />
              <Bar dataKey="Accuracy" fill="hsl(152, 55%, 35%)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="F1" fill="hsl(35, 90%, 55%)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card className="p-5 shadow-card border-border">
          <h3 className="text-sm font-semibold text-foreground uppercase tracking-wide mb-4">
            Multi-Metric Comparison
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart cx="50%" cy="50%" outerRadius="65%" data={radarData}>
              <PolarGrid stroke="hsl(145, 15%, 88%)" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "hsl(150, 10%, 45%)" }} />
              <PolarRadiusAxis tick={{ fontSize: 9 }} domain={[0, 100]} />
              {models.map((m, i) => (
                <Radar key={m.name} name={m.name} dataKey={m.name} stroke={colors[i]} fill={colors[i]} fillOpacity={0.12} />
              ))}
              <Legend wrapperStyle={{ fontSize: "10px" }} />
            </RadarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Model Architecture Cards */}
      <div>
        <h3 className="text-lg font-display font-bold text-foreground mb-4">
          Model Architectures
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {models.map((model, i) => {
            const details = MODEL_DETAILS[model.name];
            return (
              <motion.div
                key={model.name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
              >
                <Card className="p-4 shadow-soft border-border h-full">
                  <div className="flex items-start gap-3 mb-3">
                    <div className="p-2 rounded-lg bg-secondary" style={{ color: colors[i] }}>
                      {details?.icon || <Brain className="w-5 h-5" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h4 className="font-semibold text-foreground text-sm truncate">{model.name}</h4>
                        <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-secondary text-muted-foreground whitespace-nowrap">
                          {details?.type || "ML"}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                        {details?.description || "Machine learning model for plant disease detection."}
                      </p>
                    </div>
                  </div>

                  {/* Architecture pipeline */}
                  {details?.architecture && (
                    <div className="mb-3 pl-2 border-l-2 border-border ml-1 space-y-1">
                      {details.architecture.map((step, j) => (
                        <p key={j} className="text-[11px] text-muted-foreground font-mono">
                          {step}
                        </p>
                      ))}
                    </div>
                  )}

                  {/* Metrics row */}
                  <div className="grid grid-cols-4 gap-2 pt-2 border-t border-border">
                    {[
                      { label: "Acc", value: model.accuracy },
                      { label: "Prec", value: model.precision },
                      { label: "Rec", value: model.recall },
                      { label: "F1", value: model.f1Score },
                    ].map((m) => (
                      <div key={m.label} className="text-center">
                        <p className="text-xs font-bold text-foreground">
                          {m.value > 0 ? `${(m.value * 100).toFixed(1)}%` : "—"}
                        </p>
                        <p className="text-[10px] text-muted-foreground">{m.label}</p>
                      </div>
                    ))}
                  </div>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </div>
    </motion.div>
  );
};

export default ModelDashboard;
