"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Upload, ImageIcon, Sparkles, Brain, Zap, RotateCcw } from "lucide-react"
import ImageUpload from "@/components/image-upload"
import SampleGallery from "@/components/sample-gallery"
import PredictionResultComponent from "@/components/prediction-result"

interface PredictionResult {
  class: string
  confidence: number
  processing_time: number
}

export default function HomePage() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleImageSelect = (imageUrl: string) => {
    setSelectedImage(imageUrl)
    setPrediction(null)
    setError(null)
  }

  const handlePredict = async () => {
    if (!selectedImage) return

    setIsLoading(true)
    setError(null)

    try {
      // Convert image to blob if it's a data URL
      let imageBlob: Blob
      if (selectedImage.startsWith("data:")) {
        const response = await fetch(selectedImage)
        imageBlob = await response.blob()
      } else {
        const response = await fetch(selectedImage)
        imageBlob = await response.blob()
      }

      const formData = new FormData()
      formData.append("file", imageBlob, "image.jpg")

      // Replace with your FastAPI endpoint
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Prediction failed")
      }

      const result = await response.json()
      setPrediction(result)
    } catch (err) {
      setError("Unable to connect to server. Please check if FastAPI is running.")
      console.error("Prediction error:", err)
    } finally {
      setIsLoading(false)
    }
  }

  const resetSelection = () => {
    setSelectedImage(null)
    setPrediction(null)
    setError(null)
  }

  return (
    <div className="min-h-screen gradient-bg">
      <header className="border-b border-border/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/20">
                <Brain className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-balance">Animal CNN Classifier</h1>
                <p className="text-sm text-muted-foreground">AI-powered recognition of 10 animal species</p>
              </div>
            </div>
            <Badge variant="secondary" className="gap-2">
              <Sparkles className="h-4 w-4" />
              Animal AI
            </Badge>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column - Image Selection */}
          <div className="space-y-6">
            <Card className="card-gradient">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ImageIcon className="h-5 w-5 text-primary" />
                  Select Animal Image
                </CardTitle>
                <CardDescription>Upload an animal photo or choose from 10 available species</CardDescription>
              </CardHeader>
              <CardContent>
                {!selectedImage ? (
                  <Tabs defaultValue="upload" className="w-full">
                    <TabsList className="grid w-full grid-cols-2">
                      <TabsTrigger value="upload" className="gap-2">
                        <Upload className="h-4 w-4" />
                        Upload
                      </TabsTrigger>
                      <TabsTrigger value="gallery" className="gap-2">
                        <ImageIcon className="h-4 w-4" />
                        Sample Animals
                      </TabsTrigger>
                    </TabsList>

                    <TabsContent value="upload" className="mt-6">
                      <ImageUpload onImageSelect={handleImageSelect} />
                    </TabsContent>

                    <TabsContent value="gallery" className="mt-6">
                      <SampleGallery onImageSelect={handleImageSelect} />
                    </TabsContent>
                  </Tabs>
                ) : (
                  <div className="space-y-4">
                    <div className="text-center">
                      <p className="text-sm font-medium mb-4">Selected Animal Image</p>
                      <div
                        className="relative w-full max-w-sm mx-auto rounded-lg overflow-hidden bg-muted border-2 border-dashed border-primary/20"
                        style={{ aspectRatio: "1", maxHeight: "300px" }}
                      >
                        <img
                          src={selectedImage || "/placeholder.svg"}
                          alt="Selected animal"
                          className="w-full h-full object-cover"
                        />
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button onClick={handlePredict} disabled={isLoading} className="flex-1 gap-2" size="lg">
                        {isLoading ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-2 border-current border-t-transparent" />
                            Analyzing...
                          </>
                        ) : (
                          <>
                            <Zap className="h-4 w-4" />
                            Classify Animal
                          </>
                        )}
                      </Button>
                      <Button onClick={resetSelection} variant="outline" size="lg" className="gap-2 bg-transparent">
                        <RotateCcw className="h-4 w-4" />
                        Change
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Results */}
          <div className="space-y-6">
            {(prediction || isLoading || error) && (
              <Card className="card-gradient">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-accent" />
                    Classification Results
                  </CardTitle>
                  <CardDescription>Animal classification results from CNN model</CardDescription>
                </CardHeader>
                <CardContent>
                  {isLoading && (
                    <div className="text-center py-12">
                      <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent mx-auto mb-4" />
                      <p className="text-muted-foreground">Analyzing animal image...</p>
                      <Progress value={75} className="w-full mt-4" />
                    </div>
                  )}

                  {error && (
                    <div className="text-center py-12">
                      <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 mb-4">
                        <p className="text-destructive font-medium">Connection Error</p>
                        <p className="text-sm text-muted-foreground mt-1">{error}</p>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Make sure FastAPI server is running at http://localhost:8000
                      </p>
                    </div>
                  )}

                  {prediction && <PredictionResultComponent prediction={prediction} />}
                </CardContent>
              </Card>
            )}

            {!selectedImage && (
              <Card className="card-gradient">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="h-5 w-5 text-primary" />
                    Welcome to Animal Classifier
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center py-8">
                    <ImageIcon className="h-16 w-16 mx-auto mb-4 text-muted-foreground opacity-50" />
                    <p className="text-lg font-medium mb-2">Get Started</p>
                    <p className="text-muted-foreground mb-6">
                      Upload an animal photo or choose from our sample gallery to begin classification
                    </p>
                    <div className="text-sm">
                      <p className="font-medium mb-3">Supported Animal Species:</p>
                      <div className="flex flex-wrap gap-2 justify-center">
                        {[
                          "butterfly",
                          "cat",
                          "chicken",
                          "cow",
                          "dog",
                          "elephant",
                          "horse",
                          "sheep",
                          "spider",
                          "squirrel",
                        ].map((animal) => (
                          <Badge key={animal} variant="outline" className="text-xs">
                            {animal}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            
          </div>
        </div>
      </main>
    </div>
  )
}
