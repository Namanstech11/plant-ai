import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Clock, ImageIcon, Leaf } from "lucide-react";

interface LogEntry {
  id: string;
  timestamp: string;
  imageName: string;
  prediction: string;
  confidence: number;
  model: string;
}

interface PredictionLogProps {
  logs: LogEntry[];
}

const PredictionLog = ({ logs }: PredictionLogProps) => {
  if (logs.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <Leaf className="w-10 h-10 mx-auto mb-3 opacity-40" />
        <p className="text-sm">No predictions yet. Upload a leaf image to get started.</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="w-full max-w-2xl mx-auto"
    >
      <h2 className="text-2xl font-display font-bold text-foreground mb-6">
        Prediction History
      </h2>
      <div className="space-y-3">
        {logs.map((log, i) => (
          <motion.div
            key={log.id}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
          >
            <Card className="p-4 shadow-soft border-border hover:shadow-card transition-shadow">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-secondary">
                    <ImageIcon className="w-4 h-4 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-foreground">{log.prediction}</p>
                    <p className="text-xs text-muted-foreground">{log.imageName} · {log.model}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-semibold text-primary">
                    {(log.confidence * 100).toFixed(1)}%
                  </p>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock className="w-3 h-3" />
                    {log.timestamp}
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
};

export default PredictionLog;
