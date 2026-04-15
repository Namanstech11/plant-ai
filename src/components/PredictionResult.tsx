import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle, Info, Sprout } from "lucide-react";
import { Card } from "@/components/ui/card";

interface Disease {
  name: string;
  confidence: number;
  severity: "low" | "medium" | "high";
  treatment: string;
  description: string;
}

interface PredictionResultProps {
  result: Disease | null;
  allPredictions: { model: string; disease: string; confidence: number }[];
}

const severityConfig = {
  low: { icon: CheckCircle, label: "Low Severity", className: "text-success bg-success/10 border-success/20" },
  medium: { icon: Info, label: "Medium Severity", className: "text-warning bg-warning/10 border-warning/20" },
  high: { icon: AlertTriangle, label: "High Severity", className: "text-destructive bg-destructive/10 border-destructive/20" },
};

const PredictionResult = ({ result, allPredictions }: PredictionResultProps) => {
  if (!result) return null;

  const severity = severityConfig[result.severity];
  const SeverityIcon = severity.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full max-w-2xl mx-auto space-y-4"
    >
      {/* Main result */}
      <Card className="p-6 shadow-card gradient-card border-border">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl bg-primary/10">
            <Sprout className="w-7 h-7 text-primary" />
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-display font-bold text-foreground">
              {result.name}
            </h3>
            <p className="text-sm text-muted-foreground mt-1">{result.description}</p>
            <div className="flex items-center gap-3 mt-3">
              <div className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${severity.className}`}>
                <SeverityIcon className="w-3.5 h-3.5" />
                {severity.label}
              </div>
              <div className="text-sm font-semibold text-primary">
                {(result.confidence * 100).toFixed(1)}% confidence
              </div>
            </div>
          </div>
        </div>

        {/* Confidence bar */}
        <div className="mt-4">
          <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${result.confidence * 100}%` }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className="h-full gradient-leaf rounded-full"
            />
          </div>
        </div>
      </Card>

      {/* Treatment */}
      <Card className="p-5 shadow-soft border-border">
        <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide mb-2">
          {result.name.includes("Unable to confidently") ? "What to do next" : "Recommended Treatment"}
        </h4>
        <p className="text-sm text-muted-foreground leading-relaxed">
          {result.treatment}
        </p>
      </Card>

      {/* Model comparison */}
      {allPredictions.length > 0 && (
        <Card className="p-5 shadow-soft border-border">
          <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide mb-3">
            Model Predictions
          </h4>
          <div className="space-y-2">
            {allPredictions.map((pred, i) => (
              <div key={i} className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground font-medium">{pred.model}</span>
                <div className="flex items-center gap-3">
                  <span className="text-foreground">{pred.disease}</span>
                  <span className="text-xs font-mono text-primary font-semibold">
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </motion.div>
  );
};

export default PredictionResult;
