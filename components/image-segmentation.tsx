"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Camera, Upload, Loader2, ArrowRight } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { segmentImage } from "@/lib/api-client"

interface SegmentationResult {
  original_image: string
  segmented_image: string
  mask_image: string
  segments_count: number
  processing_time: number
}

export function ImageSegmentation() {
  const [image, setImage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<SegmentationResult | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleImageCapture = (imageDataUrl: string) => {
    setImage(imageDataUrl)
    setResult(null)
  }

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

  const performSegmentation = async () => {
    if (!image) return

    setIsLoading(true)
    setError(null)

    try {
      // Convert base64 to blob
      const response = await fetch(image)
      const blob = await response.blob()

      // Send to API
      const result = await segmentImage(blob)
      setResult(result)
    } catch (err) {
      setError(`Error performing segmentation: ${err instanceof Error ? err.message : String(err)}`)
      console.error("Error:", err)
    } finally {
      setIsLoading(false)
    }
  }

  const resetImage = () => {
    setImage(null)
    setResult(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Road Scene Segmentation</CardTitle>
          <CardDescription>Segment road scenes to identify different elements and surfaces</CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          {!image ? (
            <div className="space-y-6">
              <Tabs defaultValue="upload" className="mb-6">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="upload">Upload Image</TabsTrigger>
                  <TabsTrigger value="capture">Capture Image</TabsTrigger>
                </TabsList>

                <TabsContent value="upload" className="pt-4">
                  <div className="flex justify-center">
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      ref={fileInputRef}
                      onChange={handleFileUpload}
                    />
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
            </div>
          ) : (
            <div>
              {error && (
                <Alert variant="destructive" className="mb-4">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {!result ? (
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="aspect-video bg-black rounded-lg overflow-hidden">
                      <img
                        src={image || "/placeholder.svg"}
                        alt="Road Scene"
                        className="w-full h-full object-contain"
                      />
                    </div>

                    <div className="flex justify-between">
                      <Button variant="outline" onClick={resetImage}>
                        Use Different Image
                      </Button>

                      <Button onClick={performSegmentation} disabled={isLoading}>
                        {isLoading ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Processing...
                          </>
                        ) : (
                          <>
                            <Camera className="mr-2 h-4 w-4" />
                            Segment Road Scene
                          </>
                        )}
                      </Button>
                    </div>
                  </div>

                  <div className="bg-muted rounded-lg p-6 h-full flex items-center justify-center">
                    <div className="text-center">
                      <h3 className="font-medium mb-2">Ready for Segmentation</h3>
                      <p className="text-sm text-muted-foreground">
                        Click "Segment Road Scene" to analyze and segment this image
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-6">
                  <Tabs defaultValue="comparison">
                    <TabsList className="grid w-full grid-cols-3">
                      <TabsTrigger value="comparison">Before & After</TabsTrigger>
                      <TabsTrigger value="segmented">Segmented</TabsTrigger>
                      <TabsTrigger value="mask">Mask</TabsTrigger>
                    </TabsList>

                    <TabsContent value="comparison" className="mt-4">
                      <div className="flex flex-col md:flex-row items-center gap-4">
                        <div className="flex-1 aspect-video bg-black rounded-lg overflow-hidden">
                          <img
                            src={result.original_image || "/placeholder.svg"}
                            alt="Original"
                            className="w-full h-full object-contain"
                          />
                        </div>

                        <ArrowRight className="hidden md:block" />

                        <div className="flex-1 aspect-video bg-black rounded-lg overflow-hidden">
                          <img
                            src={result.segmented_image || "/placeholder.svg"}
                            alt="Segmented"
                            className="w-full h-full object-contain"
                          />
                        </div>
                      </div>
                    </TabsContent>

                    <TabsContent value="segmented" className="mt-4">
                      <div className="aspect-video bg-black rounded-lg overflow-hidden">
                        <img
                          src={result.segmented_image || "/placeholder.svg"}
                          alt="Segmented"
                          className="w-full h-full object-contain"
                        />
                      </div>
                    </TabsContent>

                    <TabsContent value="mask" className="mt-4">
                      <div className="aspect-video bg-black rounded-lg overflow-hidden">
                        <img
                          src={result.mask_image || "/placeholder.svg"}
                          alt="Mask"
                          className="w-full h-full object-contain"
                        />
                      </div>
                    </TabsContent>
                  </Tabs>

                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-muted p-4 rounded-lg">
                      <h3 className="font-medium mb-2">Segmentation Results:</h3>
                      <ul className="space-y-1">
                        <li>Segments detected: {result.segments_count}</li>
                        <li>Processing time: {result.processing_time.toFixed(2)}s</li>
                      </ul>
                    </div>

                    <div className="bg-muted p-4 rounded-lg">
                      <h3 className="font-medium mb-2">Road Elements:</h3>
                      <ul className="space-y-1">
                        <li>Road surface</li>
                        <li>Vehicles</li>
                        <li>Pedestrians</li>
                        <li>Buildings</li>
                        <li>Vegetation</li>
                      </ul>
                    </div>
                  </div>

                  <div className="flex justify-center">
                    <Button variant="outline" onClick={resetImage}>
                      Analyze Another Image
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Road Segmentation Guide</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium mb-2">What is Segmentation?</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Segmentation divides an image into different regions, identifying each pixel as belonging to a specific
                class (road, vehicle, pedestrian, etc.). This helps in understanding the composition and layout of road
                scenes.
              </p>
            </div>

            <div>
              <h3 className="font-medium mb-2">Applications</h3>
              <ul className="list-disc pl-5 space-y-1 text-sm text-muted-foreground">
                <li>Autonomous driving systems</li>
                <li>Road condition monitoring</li>
                <li>Traffic analysis</li>
                <li>Urban planning</li>
                <li>Safety assessment</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

