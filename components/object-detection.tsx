"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Camera, Upload, Loader2 } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { detectObjects } from "@/lib/api-client"

interface DetectedObject {
  label: string
  confidence: number
  bbox: number[]
}

interface DetectionResult {
  detected_objects: DetectedObject[]
  caption: string
  model_type: string
  device: string
}

export function ObjectDetection() {
  const [image, setImage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (event) => {
        if (event.target?.result) {
          setImage(event.target.result as string)
          setResult(null)
        }
      }
      reader.readAsDataURL(file)
    }
  }

  const handleImageCapture = (imageDataUrl: string) => {
    setImage(imageDataUrl)
    setResult(null)
  }

  const detectRoadObjects = async () => {
    if (!image) return

    setIsLoading(true)
    setError(null)

    try {
      // Convert base64 to blob
      const response = await fetch(image)
      const blob = await response.blob()

      // Send to API
      const data = await detectObjects(blob)
      setResult(data)

      // Draw bounding boxes
      if (canvasRef.current && image) {
        drawBoundingBoxes(data.detected_objects)
      }
    } catch (err) {
      setError(`Error detecting objects: ${err instanceof Error ? err.message : String(err)}`)
      console.error("Error:", err)
    } finally {
      setIsLoading(false)
    }
  }

  const drawBoundingBoxes = (objects: DetectedObject[]) => {
    if (!canvasRef.current || !image) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Load image to get dimensions
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.onload = () => {
      // Set canvas dimensions to match image
      canvas.width = img.width
      canvas.height = img.height

      // Draw the image
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

      // Draw bounding boxes
      objects.forEach((obj) => {
        const [x1, y1, x2, y2] = obj.bbox
        const width = x2 - x1
        const height = y2 - y1

        // Color based on object type
        let color = "red"
        if (obj.label.includes("car") || obj.label.includes("truck") || obj.label.includes("bus")) {
          color = "blue"
        } else if (obj.label.includes("person") || obj.label.includes("pedestrian")) {
          color = "green"
        } else if (obj.label.includes("sign") || obj.label.includes("light")) {
          color = "orange"
        }

        // Draw rectangle
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.strokeRect(x1, y1, width, height)

        // Draw label
        ctx.fillStyle = `${color}cc`
        ctx.fillRect(x1, y1 - 20, 120, 20)
        ctx.fillStyle = "white"
        ctx.font = "12px Arial"
        ctx.fillText(`${obj.label} ${Math.round(obj.confidence * 100)}%`, x1 + 5, y1 - 5)
      })
    }
    img.src = image
  }

  const resetImage = () => {
    setImage(null)
    setResult(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const getObjectCounts = () => {
    if (!result) return {}

    const counts: Record<string, number> = {}
    result.detected_objects.forEach((obj) => {
      const category = getObjectCategory(obj.label)
      counts[category] = (counts[category] || 0) + 1
    })

    return counts
  }

  const getObjectCategory = (label: string) => {
    if (label.includes("car") || label.includes("truck") || label.includes("bus") || label.includes("motorcycle")) {
      return "Vehicles"
    } else if (label.includes("person") || label.includes("pedestrian")) {
      return "Pedestrians"
    } else if (label.includes("sign") || label.includes("light")) {
      return "Traffic Signs/Lights"
    } else {
      return "Other"
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Road Object Detection</CardTitle>
          <CardDescription>Analyze road scenes to detect vehicles, pedestrians, and traffic elements</CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          <Tabs defaultValue="upload" className="mb-6">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="upload">Upload Image</TabsTrigger>
              <TabsTrigger value="capture">Capture Image</TabsTrigger>
            </TabsList>

            <TabsContent value="upload" className="pt-4">
              <div className="flex justify-center">
                <input type="file" accept="image/*" className="hidden" ref={fileInputRef} onChange={handleFileUpload} />
                <Button onClick={() => fileInputRef.current?.click()}>
                  <Upload className="mr-2 h-4 w-4" />
                  Upload Road Scene Image
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="capture" className="pt-4">
              <div className="flex justify-center">
                <div className="w-full max-w-md">
                  <Card className="border-0 shadow-none">
                    <CardContent className="p-0">
                      <div className="aspect-video bg-black rounded-lg overflow-hidden">
                        <Camera className="w-full h-full text-muted-foreground" />
                      </div>
                      <Button className="w-full mt-4" onClick={() => {}}>
                        <Camera className="mr-2 h-4 w-4" />
                        Capture Road Scene
                      </Button>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>
          </Tabs>

          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {image && (
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="aspect-video bg-black rounded-lg overflow-hidden relative">
                  {result ? (
                    <canvas ref={canvasRef} className="w-full h-full object-contain" />
                  ) : (
                    <img src={image || "/placeholder.svg"} alt="Road Scene" className="w-full h-full object-contain" />
                  )}
                </div>

                <div className="flex justify-between">
                  <Button variant="outline" onClick={resetImage}>
                    Use Different Image
                  </Button>

                  <Button onClick={detectRoadObjects} disabled={isLoading || !image}>
                    {isLoading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Camera className="mr-2 h-4 w-4" />
                        Detect Road Objects
                      </>
                    )}
                  </Button>
                </div>
              </div>

              <div className="space-y-4">
                {result && (
                  <>
                    <div className="bg-muted p-4 rounded-lg">
                      <h3 className="font-medium mb-2">Scene Description:</h3>
                      <p>{result.caption}</p>
                    </div>

                    <div>
                      <h3 className="font-medium mb-2">Object Summary:</h3>
                      <div className="flex flex-wrap gap-2 mb-4">
                        {Object.entries(getObjectCounts()).map(([category, count]) => (
                          <Badge key={category} variant="outline" className="text-sm">
                            {category}: {count}
                          </Badge>
                        ))}
                      </div>

                      <h3 className="font-medium mb-2">Detected Objects:</h3>
                      <div className="max-h-[300px] overflow-y-auto">
                        <ul className="space-y-2">
                          {result.detected_objects.map((obj, index) => (
                            <li key={index} className="bg-muted p-2 rounded flex justify-between">
                              <span>{obj.label}</span>
                              <span className="text-muted-foreground">{Math.round(obj.confidence * 100)}%</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <div className="text-xs text-muted-foreground mt-4">
                      <p>Model: {result.model_type}</p>
                      <p>Device: {result.device}</p>
                    </div>
                  </>
                )}

                {!result && image && (
                  <div className="bg-muted rounded-lg p-6 h-full flex items-center justify-center">
                    <div className="text-center">
                      <h3 className="font-medium mb-2">Ready for Analysis</h3>
                      <p className="text-sm text-muted-foreground">
                        Click "Detect Road Objects" to analyze this road scene
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {!image && (
            <div className="bg-muted rounded-lg p-8 text-center">
              <h3 className="font-medium mb-2">No Image Selected</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Upload or capture a road scene image to begin detection
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Road Object Detection Guide</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium mb-2">Best Practices</h3>
              <ul className="list-disc pl-5 space-y-2">
                <li>Use clear, well-lit images of road scenes</li>
                <li>Ensure the image captures the full road environment</li>
                <li>For best results, avoid extreme weather conditions</li>
                <li>Images with multiple road elements provide richer analysis</li>
              </ul>
            </div>

            <div>
              <h3 className="font-medium mb-2">Detectable Objects</h3>
              <ul className="list-disc pl-5 space-y-2">
                <li>
                  <span className="font-medium">Vehicles:</span> Cars, trucks, buses, motorcycles
                </li>
                <li>
                  <span className="font-medium">People:</span> Pedestrians, cyclists
                </li>
                <li>
                  <span className="font-medium">Infrastructure:</span> Traffic lights, road signs
                </li>
                <li>
                  <span className="font-medium">Obstacles:</span> Various road obstacles
                </li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

