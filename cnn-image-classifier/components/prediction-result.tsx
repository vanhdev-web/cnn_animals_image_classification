"use client"

import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Card, CardContent } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Trophy, Clock, Target, TrendingUp } from "lucide-react"

interface PredictionResult {
  class: string
  confidence: number
  processing_time: number
  top_predictions?: Array<{
    class: string
    confidence: number
  }>
}

interface PredictionResultProps {
  prediction: PredictionResult
}

export default function PredictionResultComponent({ prediction }: PredictionResultProps) {
  const confidencePercentage = Math.round(prediction.confidence * 100)
  const processingTimeMs = Math.round(prediction.processing_time * 1000)

  // Mock top predictions if not provided
  const topPredictions = prediction.top_predictions || [
    { class: prediction.class, confidence: prediction.confidence },
    { class: "Alternative 1", confidence: prediction.confidence * 0.7 },
    { class: "Alternative 2", confidence: prediction.confidence * 0.4 },
  ]

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-400"
    if (confidence >= 0.6) return "text-yellow-400"
    return "text-orange-400"
  }

  const getConfidenceBadgeVariant = (confidence: number) => {
    if (confidence >= 0.8) return "default"
    if (confidence >= 0.6) return "secondary"
    return "outline"
  }

  return (
    <div className="space-y-6">
      {/* Main Result */}
      <Card className="card-gradient border-primary/20">
        <CardContent className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Trophy className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Kết quả chính</h3>
            </div>
            <Badge variant={getConfidenceBadgeVariant(prediction.confidence)} className="gap-1">
              <Target className="h-3 w-3" />
              {confidencePercentage}%
            </Badge>
          </div>

          <div className="text-center space-y-4">
            <div>
              <h2 className="text-3xl font-bold text-primary mb-2">{prediction.class}</h2>
              <p className="text-muted-foreground">Phân loại được dự đoán</p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Độ tin cậy</span>
                <span className={getConfidenceColor(prediction.confidence)}>{confidencePercentage}%</span>
              </div>
              <Progress value={confidencePercentage} className="h-3" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Top Predictions */}
      <Card className="card-gradient">
        <CardContent className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="h-5 w-5 text-accent" />
            <h3 className="font-semibold">Top dự đoán</h3>
          </div>

          <div className="space-y-3">
            {topPredictions.slice(0, 3).map((pred, index) => (
              <div key={index} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        index === 0 ? "bg-primary" : index === 1 ? "bg-accent" : "bg-muted-foreground"
                      }`}
                    />
                    <span className="font-medium">{pred.class}</span>
                  </div>
                  <span className="text-sm text-muted-foreground">{Math.round(pred.confidence * 100)}%</span>
                </div>
                <Progress value={pred.confidence * 100} className="h-2" />
                {index < topPredictions.length - 1 && <Separator className="mt-3" />}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Processing Stats */}
      <Card className="card-gradient">
        <CardContent className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <Clock className="h-5 w-5 text-muted-foreground" />
            <h3 className="font-semibold">Thống kê xử lý</h3>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-4 rounded-lg bg-muted/50">
              <div className="text-2xl font-bold text-primary mb-1">{processingTimeMs}ms</div>
              <div className="text-xs text-muted-foreground">Thời gian xử lý</div>
            </div>

            <div className="text-center p-4 rounded-lg bg-muted/50">
              <div className="text-2xl font-bold text-accent mb-1">CNN</div>
              <div className="text-xs text-muted-foreground">Mô hình AI</div>
            </div>
          </div>

          <Separator className="my-4" />

          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Trạng thái:</span>
              <Badge variant="outline" className="text-green-400 border-green-400/20">
                Thành công
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Timestamp:</span>
              <span className="text-xs font-mono">{new Date().toLocaleTimeString("vi-VN")}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Confidence Interpretation */}
      <Card className="card-gradient">
        <CardContent className="p-4">
          <div className="text-sm space-y-2">
            <h4 className="font-medium mb-3">Giải thích độ tin cậy:</h4>
            <div className="space-y-1 text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-400" />
                <span>≥ 80%: Rất tin cậy</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-yellow-400" />
                <span>60-79%: Tin cậy vừa phải</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-orange-400" />
                <span>{"< 60%"}: Cần xem xét thêm</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
