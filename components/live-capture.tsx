"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { RefreshCw, Zap, Camera, Video } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"

interface DetectedObject {
  label: string
  confidence: number
  bbox: [number, number, number, number]
}

export function LiveCapture() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("")
  const [error, setError] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([])
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null)
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [recordingTimer, setRecordingTimer] = useState<NodeJS.Timeout | null>(null)
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  const [confidenceThreshold, setConfidenceThreshold] = useState(50)
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([])
  const [fps, setFps] = useState(0)
  const fpsCounterRef = useRef<number>(0)
  const lastTimeRef = useRef<number>(performance.now())
  const animationRef = useRef<number | null>(null)
  const processingIntervalRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    if (typeof window !== 'undefined') {
      getDevices()
      return () => {
        if (stream) {
          stream.getTracks().forEach((track) => track.stop())
        }
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current)
        }
        if (processingIntervalRef.current) {
          clearInterval(processingIntervalRef.current)
        }
        if (recordingTimer) {
          clearInterval(recordingTimer)
        }
      }
    }
  }, [stream, recordingTimer])

  useEffect(() => {
    if (selectedDeviceId) {
      startCamera()
    }
  }, [selectedDeviceId])

  useEffect(() => {
    if (isProcessing && stream) {
      startObjectDetection()
    } else {
      stopObjectDetection()
    }
  }, [isProcessing, stream, showBoundingBoxes, showLabels, confidenceThreshold])

  const getDevices = async () => {
    if (typeof window === 'undefined') return
    try {
      // First request camera access to get permission
      const initialStream = await navigator.mediaDevices.getUserMedia({ video: true })
      initialStream.getTracks().forEach((track) => track.stop())

      // Now get the list of devices
      const devices = await navigator.mediaDevices.enumerateDevices()
      const videoDevices = devices.filter((device) => device.kind === "videoinput")
      setDevices(videoDevices)

      if (videoDevices.length > 0) {
        setSelectedDeviceId(videoDevices[0].deviceId)
      }
    } catch (err) {
      setError("Error accessing camera devices. Please make sure you've granted camera permissions.")
      console.error("Error accessing devices:", err)
    }
  }

  const startCamera = async () => {
    try {
      // Stop any existing stream
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
        },
        audio: true,
      })
      setStream(mediaStream)

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }

      // Initialize media recorder
      const recorder = new MediaRecorder(mediaStream)
      setMediaRecorder(recorder)

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setRecordedChunks((prev) => [...prev, event.data])
        }
      }

      recorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: "video/webm" })
        const url = URL.createObjectURL(blob)

        // Download the recorded video
        const a = document.createElement("a")
        a.href = url
        a.download = `road-recording-${new Date().getTime()}.webm`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)

        // Reset recording state
        setRecordedChunks([])
      }

      setError(null)
    } catch (err) {
      setError("Error accessing camera. Please make sure you've granted camera permissions.")
      console.error("Error accessing camera:", err)
    }
  }

  const toggleProcessing = () => {
    setIsProcessing(!isProcessing)
  }

  const startObjectDetection = () => {
    // Clear any existing interval
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current)
    }

    // Start processing frames at regular intervals
    processingIntervalRef.current = setInterval(() => {
      processFrame()
    }, 100) // Process every 100ms

    // Start animation loop for rendering
    renderFrame()
  }

  const stopObjectDetection = () => {
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current)
      processingIntervalRef.current = null
    }

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }
  }

  const processFrame = () => {
    try {
      if (!videoRef.current || !canvasRef.current || !stream) {
        console.warn("Required refs not available for processing frame")
        return
      }

      // Simulate object detection with random objects
      // In a real implementation, this would call the API or run a local model
      const mockObjects: DetectedObject[] = []

      // Generate 0-5 random objects
      const objectCount = Math.floor(Math.random() * 6)
      const possibleObjects = ["car", "truck", "pedestrian", "bicycle", "traffic light", "stop sign", "bus"]

      for (let i = 0; i < objectCount; i++) {
        const label = possibleObjects[Math.floor(Math.random() * possibleObjects.length)]
        const confidence = Math.random() * 0.5 + 0.5 // 0.5-1.0

        // Random position within the frame
        const x1 = Math.random() * (videoRef.current.videoWidth * 0.7)
        const y1 = Math.random() * (videoRef.current.videoHeight * 0.7)
        const width = Math.random() * 100 + 50
        const height = Math.random() * 100 + 50

        mockObjects.push({
          label,
          confidence,
          bbox: [x1, y1, x1 + width, y1 + height],
        })
      }

      setDetectedObjects(mockObjects)

      // Update FPS counter
      fpsCounterRef.current++
      const now = performance.now()
      if (now - lastTimeRef.current >= 1000) {
        setFps(fpsCounterRef.current)
        fpsCounterRef.current = 0
        lastTimeRef.current = now
      }
    } catch (err) {
      console.error('Error processing frame:', err)
      setError('Error processing frame')
    }
  }

  const renderFrame = () => {
    try {
      if (!videoRef.current || !canvasRef.current || !stream) {
        console.warn("Required refs not available for rendering frame")
        return
      }

      const video = videoRef.current
      const canvas = canvasRef.current
      const context = canvas.getContext("2d")

      if (!context) {
        console.error("Could not get canvas context")
        return
      }

      // Set canvas dimensions to match video
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      // Draw the current video frame to the canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height)

      // Draw bounding boxes if enabled
      if (showBoundingBoxes) {
        detectedObjects.forEach((obj) => {
          if (obj.confidence * 100 >= confidenceThreshold) {
            const [x1, y1, x2, y2] = obj.bbox
            const width = x2 - x1
            const height = y2 - y1

            // Draw rectangle
            context.strokeStyle = getColorForLabel(obj.label)
            context.lineWidth = 2
            context.strokeRect(x1, y1, width, height)

            // Draw label if enabled
            if (showLabels) {
              const labelText = `${obj.label} ${Math.round(obj.confidence * 100)}%`
              const textWidth = context.measureText(labelText).width

              context.fillStyle = getColorForLabel(obj.label, 0.7)
              context.fillRect(x1, y1 - 20, textWidth + 10, 20)
              context.fillStyle = "white"
              context.font = "12px Arial"
              context.fillText(labelText, x1 + 5, y1 - 5)
            }
          }
        })
      }

      // Continue the animation loop
      animationRef.current = requestAnimationFrame(renderFrame)
    } catch (err) {
      console.error('Error rendering frame:', err)
      setError('Error rendering frame')
    }
  }

  const getColorForLabel = (label: string, alpha = 1) => {
    const colors: Record<string, string> = {
      car: `rgba(255, 0, 0, ${alpha})`,
      truck: `rgba(0, 255, 0, ${alpha})`,
      pedestrian: `rgba(0, 0, 255, ${alpha})`,
      bicycle: `rgba(255, 255, 0, ${alpha})`,
      "traffic light": `rgba(255, 0, 255, ${alpha})`,
      "stop sign": `rgba(0, 255, 255, ${alpha})`,
      bus: `rgba(255, 165, 0, ${alpha})`,
    }

    return colors[label] || `rgba(128, 128, 128, ${alpha})`
  }

  const startRecording = () => {
    if (!mediaRecorder || !stream) return

    setRecordedChunks([])
    setRecordingDuration(0)

    mediaRecorder.start(100) // Collect data every 100ms
    setIsRecording(true)

    // Start timer
    const timer = setInterval(() => {
      setRecordingDuration((prev) => prev + 1)
    }, 1000)
    setRecordingTimer(timer)
  }

  const stopRecording = () => {
    if (!mediaRecorder || !isRecording) return

    mediaRecorder.stop()
    setIsRecording(false)

    // Stop timer
    if (recordingTimer) {
      clearInterval(recordingTimer)
      setRecordingTimer(null)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Live Road Scene Analysis</CardTitle>
          <CardDescription>Real-time object detection for road scenes</CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="grid md:grid-cols-3 gap-6">
            <div className="md:col-span-2 space-y-4">
              <div className="mb-4">
                <Select value={selectedDeviceId} onValueChange={setSelectedDeviceId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select camera" />
                  </SelectTrigger>
                  <SelectContent>
                    {devices.map((device) => (
                      <SelectItem key={device.deviceId} value={device.deviceId}>
                        {device.label || `Camera ${devices.indexOf(device) + 1}`}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={isProcessing ? "hidden" : "w-full h-full object-cover"}
                />
                {isProcessing && (
                  <canvas ref={canvasRef} className="w-full h-full object-cover" />
                )}

                {isProcessing && (
                  <div className="absolute top-2 left-2 bg-black/70 text-white px-2 py-1 rounded text-xs">
                    FPS: {fps}
                  </div>
                )}

                {isRecording && (
                  <div className="absolute top-2 right-2 bg-red-500 text-white px-2 py-1 rounded-md flex items-center text-xs">
                    <div className="w-2 h-2 rounded-full bg-white mr-2 animate-pulse"></div>
                    REC {formatTime(recordingDuration)}
                  </div>
                )}
              </div>

              <div className="flex flex-wrap gap-2">
                <Button variant={isProcessing ? "default" : "outline"} onClick={toggleProcessing}>
                  <Zap className="mr-2 h-4 w-4" />
                  {isProcessing ? "Processing Live" : "Start Processing"}
                </Button>

                <Button variant="ghost" onClick={startCamera}>
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Restart Camera
                </Button>

                {!isRecording ? (
                  <Button variant="outline" onClick={startRecording} disabled={!stream}>
                    <Video className="mr-2 h-4 w-4" />
                    Record
                  </Button>
                ) : (
                  <Button variant="destructive" onClick={stopRecording}>
                    <Camera className="mr-2 h-4 w-4" />
                    Stop Recording
                  </Button>
                )}
              </div>
            </div>

            <div className="space-y-6">
              <div className="space-y-4">
                <h3 className="font-medium">Detection Settings</h3>

                <div className="flex items-center justify-between">
                  <Label htmlFor="show-boxes">Show Bounding Boxes</Label>
                  <Switch id="show-boxes" checked={showBoundingBoxes} onCheckedChange={setShowBoundingBoxes} />
                </div>

                <div className="flex items-center justify-between">
                  <Label htmlFor="show-labels">Show Labels</Label>
                  <Switch id="show-labels" checked={showLabels} onCheckedChange={setShowLabels} />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="confidence">Confidence Threshold</Label>
                    <span className="text-sm">{confidenceThreshold}%</span>
                  </div>
                  <Slider
                    id="confidence"
                    value={[confidenceThreshold]}
                    min={0}
                    max={100}
                    step={5}
                    onValueChange={(values) => setConfidenceThreshold(values[0])}
                  />
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="font-medium">Detected Objects</h3>

                {isProcessing ? (
                  detectedObjects.length > 0 ? (
                    <div className="space-y-2 max-h-[200px] overflow-y-auto">
                      {detectedObjects
                        .filter((obj) => obj.confidence * 100 >= confidenceThreshold)
                        .map((obj, idx) => (
                          <div key={idx} className="flex justify-between items-center bg-muted p-2 rounded">
                            <div className="flex items-center gap-2">
                              <div
                                className="w-3 h-3 rounded-full"
                                style={{ backgroundColor: getColorForLabel(obj.label) }}
                              ></div>
                              <span>{obj.label}</span>
                            </div>
                            <Badge variant="outline">{Math.round(obj.confidence * 100)}%</Badge>
                          </div>
                        ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No objects detected</p>
                  )
                ) : (
                  <p className="text-sm text-muted-foreground">Start processing to detect objects</p>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Live Detection Tips</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc pl-5 space-y-2">
            <li>Position your camera with a clear view of the road</li>
            <li>Adjust the confidence threshold based on detection quality</li>
            <li>Record important moments for later analysis</li>
            <li>For best performance, ensure good lighting conditions</li>
            <li>Higher FPS indicates better real-time performance</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}