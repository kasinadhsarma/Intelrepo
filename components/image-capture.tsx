"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Camera, Download, RefreshCw, Upload, ArrowRight } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { detectObjects } from "@/lib/api-client"

interface ImageCaptureProps {
  onCapture?: (imageDataUrl: string) => void
  hideDownload?: boolean
}

export function ImageCapture({ onCapture, hideDownload = false }: ImageCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [processingResult, setProcessingResult] = useState<any>(null)
  const [isProcessing, setIsProcessing] = useState(false)

  useEffect(() => {
    // Only run on client-side
    if (typeof window !== 'undefined') {
      startCamera()
      return () => {
        if (stream) {
          stream.getTracks().forEach((track) => track.stop())
        }
      }
    }
  }, [stream])

  const startCamera = async () => {
    if (typeof window === 'undefined') return
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
      })
      setStream(mediaStream)

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }

      setError(null)
    } catch (err) {
      setError("Error accessing camera. Please make sure you've granted camera permissions.")
      console.error("Error accessing camera:", err)
    }
  }

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current
      const canvas = canvasRef.current
      const context = canvas.getContext("2d")

      // Set canvas dimensions to match video
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      // Draw the current video frame to the canvas
      context?.drawImage(video, 0, 0, canvas.width, canvas.height)

      // Convert canvas to data URL
      const imageDataUrl = canvas.toDataURL("image/png")
      setCapturedImage(imageDataUrl)
      setUploadedImage(null)

      // Call the onCapture callback if provided
      if (onCapture) {
        onCapture(imageDataUrl)
      }
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (event) => {
        if (event.target?.result) {
          setUploadedImage(event.target.result as string)
          setCapturedImage(null)

          // Call the onCapture callback if provided
          if (onCapture) {
            onCapture(event.target.result as string)
          }
        }
      }
      reader.readAsDataURL(file)
    }
  }

  const resetCapture = () => {
    setCapturedImage(null)
    setUploadedImage(null)
    setProcessingResult(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const downloadImage = () => {
    const imageToDownload = capturedImage || uploadedImage
    if (imageToDownload) {
      const link = document.createElement("a")
      link.href = imageToDownload
      link.download = `road-image-${new Date().getTime()}.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  const processImage = async () => {
    const imageToProcess = capturedImage || uploadedImage
    if (!imageToProcess) return

    setIsProcessing(true)
    try {
      // Convert base64 to blob
      const response = await fetch(imageToProcess)
      const blob = await response.blob()

      // Send to API
      const result = await detectObjects(blob)
      setProcessingResult(result)
    } catch (err) {
      setError(`Error processing image: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setIsProcessing(false)
    }
  }

  const currentImage = capturedImage || uploadedImage

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Road Scene Image Capture</CardTitle>
          <CardDescription>Capture or upload images of road scenes for analysis</CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              {!currentImage ? (
                <>
                  <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                    <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                  </div>

                  <div className="flex justify-center gap-4">
                    <Button onClick={captureImage}>
                      <Camera className="mr-2 h-4 w-4" />
                      Capture Image
                    </Button>

                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      ref={fileInputRef}
                      onChange={handleFileUpload}
                    />
                    <Button variant="outline" onClick={() => fileInputRef.current?.click()}>
                      <Upload className="mr-2 h-4 w-4" />
                      Upload Image
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                    <img
                      src={currentImage || "/placeholder.svg"}
                      alt="Captured"
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <canvas ref={canvasRef} className="hidden" />

                  <div className="flex justify-center gap-4">
                    <Button variant="outline" onClick={resetCapture}>
                      <RefreshCw className="mr-2 h-4 w-4" />
                      New Image
                    </Button>

                    {!hideDownload && (
                      <Button onClick={downloadImage}>
                        <Download className="mr-2 h-4 w-4" />
                        Download
                      </Button>
                    )}

                    {!onCapture && (
                      <Button onClick={processImage} disabled={isProcessing}>
                        <ArrowRight className="mr-2 h-4 w-4" />
                        {isProcessing ? "Processing..." : "Analyze"}
                      </Button>
                    )}
                  </div>
                </>
              )}
            </div>

            {processingResult && (
              <div className="space-y-4">
                <h3 className="font-medium text-lg">Analysis Results</h3>

                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Caption:</h4>
                  <p>{processingResult.caption}</p>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Detected Objects:</h4>
                  <ul className="space-y-2">
                    {processingResult.detected_objects.map((obj: any, index: number) => (
                      <li key={index} className="bg-muted p-2 rounded flex justify-between">
                        <span>{obj.label}</span>
                        <span className="text-muted-foreground">{Math.round(obj.confidence * 100)}%</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Tips for Road Scene Capture</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc pl-5 space-y-2">
            <li>Position your camera to capture the full road scene</li>
            <li>Ensure good lighting conditions for better detection</li>
            <li>Try to capture scenes with various road elements (vehicles, signs, pedestrians)</li>
            <li>For best results, keep the camera steady</li>
            <li>Avoid capturing in extreme weather conditions unless specifically testing those scenarios</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
