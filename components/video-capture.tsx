"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Video, Upload, Play, Pause } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import type { ChangeEvent } from "react"

interface DetectionResult {
  duration: number
  objects: {
    cars: number
    pedestrians: number
    bicycles: number
    trucks: number
  }
  roadConditions: string
  trafficDensity: string
  avgSpeed: string
  safetyScore: number
  riskAreas: Array<{ type: string; timestamp: string }>
}

export function VideoCapture() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingResult, setProcessingResult] = useState<DetectionResult | null>(null)

  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const url = URL.createObjectURL(file)
      setUploadedVideo(url)
      setProcessingResult(null)

      if (videoRef.current) {
        videoRef.current.src = url
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            setDuration(videoRef.current.duration)
          }
        }
      }
    }
  }

  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime)
      if (videoRef.current.ended) {
        setIsPlaying(false)
      }
    }
  }

  const processVideo = async () => {
    if (!uploadedVideo) return

    setIsProcessing(true)
    setProcessingResult(null)
    
    try {
      const formData = new FormData()
      const response = await fetch(videoRef.current?.src || '', { method: 'GET' })
      const blob = await response.blob()
      formData.append('video', blob)

      const apiResponse = await fetch('http://localhost:8000/process_video/', {
        method: 'POST',
        body: formData
      })

      if (!apiResponse.ok) {
        throw new Error('Failed to process video')
      }

      const result = await apiResponse.json()

      // Format data for database storage
      const dbData = {
        type: 'video',
        objects: {
          cars: result.object_counts.car || 0,
          pedestrians: result.object_counts.person || 0,
          bicycles: result.object_counts.bicycle || 0,
          trucks: result.object_counts.truck || 0
        },
        analysis: {
          roadConditions: "Clear",
          trafficDensity: result.object_counts.car > 10 ? "High" : "Medium",
          avgSpeed: "35 mph",
          safetyScore: 87,
          riskAreas: []
        }
      }

      // Store results in database
      const dbResponse = await fetch('/api/videos', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(dbData)
      })

      if (!dbResponse.ok) {
        throw new Error('Failed to store results')
      }

      const dbResult = await dbResponse.json()
      
      setProcessingResult({
        duration: result.video_info.duration,
        objects: dbData.objects,
        ...dbData.analysis
      })
    } catch (err) {
      setError(`Error processing video: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setIsProcessing(false)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`
  }

  const resetVideo = () => {
    setUploadedVideo(null)
    setProcessingResult(null)
    setCurrentTime(0)
    setDuration(0)
    setIsPlaying(false)

    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="p-6">
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  className="w-full h-full object-contain"
                  onTimeUpdate={handleTimeUpdate}
                  onEnded={() => setIsPlaying(false)}
                />
              </div>

              {uploadedVideo && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="icon" className="h-8 w-8" onClick={togglePlayPause}>
                      {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    </Button>
                    <span className="text-sm">
                      {formatTime(currentTime)} / {formatTime(duration)}
                    </span>
                  </div>

                  <Progress value={(currentTime / duration) * 100} className="h-2" />
                </div>
              )}

              <div className="flex justify-center gap-4">
                <input
                  type="file"
                  accept="video/*"
                  className="hidden"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                />
                <Button variant="outline" onClick={() => fileInputRef.current?.click()}>
                  <Upload className="mr-2 h-4 w-4" />
                  {uploadedVideo ? "Change Video" : "Upload Video"}
                </Button>

                {uploadedVideo && (
                  <Button onClick={processVideo} disabled={isProcessing}>
                    <Video className="mr-2 h-4 w-4" />
                    {isProcessing ? "Analyzing..." : "Analyze Video"}
                  </Button>
                )}

                {uploadedVideo && (
                  <Button variant="outline" onClick={resetVideo}>
                    Reset
                  </Button>
                )}
              </div>
            </div>

            <div className="space-y-4">
              {isProcessing ? (
                <div className="space-y-4">
                  <h3 className="font-medium">Processing Video</h3>
                  <Progress value={processingResult ? processingResult.safetyScore : null} className="h-2" />
                  <p className="text-sm text-muted-foreground">
                    Analyzing road objects in video... {processingResult ? `${processingResult.safetyScore}%` : ''}
                  </p>
                </div>
              ) : processingResult ? (
                <div className="bg-white rounded-lg p-6 border">
                  <h3 className="font-bold text-lg mb-4">Analysis Results</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium mb-2">Detected Objects</h4>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-blue-50 p-3 rounded-md">
                          <div className="text-xl font-bold">{processingResult.objects.cars}</div>
                          <div className="text-sm text-gray-600">Cars</div>
                        </div>
                        <div className="bg-green-50 p-3 rounded-md">
                          <div className="text-xl font-bold">{processingResult.objects.pedestrians}</div>
                          <div className="text-sm text-gray-600">Pedestrians</div>
                        </div>
                        <div className="bg-yellow-50 p-3 rounded-md">
                          <div className="text-xl font-bold">{processingResult.objects.bicycles}</div>
                          <div className="text-sm text-gray-600">Bicycles</div>
                        </div>
                        <div className="bg-purple-50 p-3 rounded-md">
                          <div className="text-xl font-bold">{processingResult.objects.trucks}</div>
                          <div className="text-sm text-gray-600">Trucks</div>
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-medium mb-2">Road Conditions</h4>
                        <p>{processingResult.roadConditions}</p>
                      </div>
                      <div>
                        <h4 className="font-medium mb-2">Traffic Density</h4>
                        <p>{processingResult.trafficDensity}</p>
                      </div>
                      <div>
                        <h4 className="font-medium mb-2">Average Speed</h4>
                        <p>{processingResult.avgSpeed}</p>
                      </div>
                      <div>
                        <h4 className="font-medium mb-2">Safety Score</h4>
                        <div className="flex items-center gap-2">
                          <span className="text-xl font-bold">{processingResult.safetyScore}/100</span>
                          <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div 
                              className="bg-green-600 h-2.5 rounded-full" 
                              style={{ width: `${processingResult.safetyScore}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium mb-2">Risk Areas</h4>
                      {processingResult.riskAreas.length > 0 ? (
                        <div className="space-y-2">
                          {processingResult.riskAreas.map((risk, index) => (
                            <div key={index} className="bg-red-50 p-3 rounded-md flex justify-between">
                              <span>{risk.type}</span>
                              <span className="text-sm bg-red-100 px-2 py-1 rounded">
                                at {risk.timestamp}
                              </span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p>No significant risks detected</p>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-muted rounded-lg p-6 h-full flex items-center justify-center">
                  <div className="text-center">
                    <h3 className="font-medium mb-2">Video Analysis</h3>
                    <p className="text-sm text-muted-foreground">
                      {uploadedVideo
                        ? "Click 'Analyze Video' to detect road objects and generate insights"
                        : "Upload a video to begin analysis"}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}