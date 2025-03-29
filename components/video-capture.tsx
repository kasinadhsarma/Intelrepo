"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Video, Square, Download, RefreshCw, Upload, Play, Pause } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"

export function VideoCapture() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const recordedVideoRef = useRef<HTMLVideoElement>(null)
  const uploadedVideoRef = useRef<HTMLVideoElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null)
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([])
  const [isRecording, setIsRecording] = useState(false)
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [recordingTimer, setRecordingTimer] = useState<NodeJS.Timeout | null>(null)
  const [recordedVideo, setRecordedVideo] = useState<string | null>(null)
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)

  useEffect(() => {
    // Only run on client-side
    if (typeof window !== 'undefined') {
      startCamera()
      return () => {
        if (stream) {
          stream.getTracks().forEach((track) => track.stop())
        }
        if (recordingTimer) {
          clearInterval(recordingTimer)
        }
      }
    }
  }, [stream, recordingTimer])

  const startCamera = async () => {
    if (typeof window === 'undefined') return
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      })
      setStream(mediaStream)

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }

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
        setRecordedVideo(url)
        setUploadedVideo(null)

        if (recordedVideoRef.current) {
          recordedVideoRef.current.src = url
          recordedVideoRef.current.onloadedmetadata = () => {
            if (recordedVideoRef.current) {
              setDuration(recordedVideoRef.current.duration)
            }
          }
        }
      }

      setError(null)
    } catch (err) {
      setError("Error accessing camera. Please make sure you've granted camera permissions.")
      console.error("Error accessing camera:", err)
    }
  }

  const startRecording = () => {
    setRecordedChunks([])
    setRecordedVideo(null)
    setRecordingDuration(0)

    if (mediaRecorder) {
      mediaRecorder.start(100) // Collect data every 100ms
      setIsRecording(true)

      // Start timer
      const timer = setInterval(() => {
        setRecordingDuration((prev) => prev + 1)
      }, 1000)
      setRecordingTimer(timer)
    }
  }

  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop()
      setIsRecording(false)

      // Stop timer
      if (recordingTimer) {
        clearInterval(recordingTimer)
        setRecordingTimer(null)
      }
    }
  }

  const resetRecording = () => {
    setRecordedVideo(null)
    setUploadedVideo(null)
    setRecordedChunks([])
    setCurrentTime(0)
    setDuration(0)
    setIsPlaying(false)

    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const url = URL.createObjectURL(file)
      setUploadedVideo(url)
      setRecordedVideo(null)

      if (uploadedVideoRef.current) {
        uploadedVideoRef.current.src = url
        uploadedVideoRef.current.onloadedmetadata = () => {
          if (uploadedVideoRef.current) {
            setDuration(uploadedVideoRef.current.duration)
          }
        }
      }
    }
  }

  const togglePlayPause = () => {
    const videoElement = recordedVideo ? recordedVideoRef.current : uploadedVideoRef.current

    if (videoElement) {
      if (isPlaying) {
        videoElement.pause()
      } else {
        videoElement.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const handleTimeUpdate = () => {
    const videoElement = recordedVideo ? recordedVideoRef.current : uploadedVideoRef.current

    if (videoElement) {
      setCurrentTime(videoElement.currentTime)

      if (videoElement.ended) {
        setIsPlaying(false)
      }
    }
  }

  const downloadVideo = () => {
    const videoToDownload = recordedVideo || uploadedVideo

    if (!videoToDownload) return

    const a = document.createElement("a")
    a.href = videoToDownload
    a.download = `road-video-${new Date().getTime()}.${recordedVideo ? "webm" : "mp4"}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const processVideo = async () => {
    const videoToProcess = recordedVideo || uploadedVideo
    if (!videoToProcess) return

    setIsProcessing(true)

    // Simulate processing progress
    let progress = 0
    const interval = setInterval(() => {
      progress += 5
      setProcessingProgress(progress)

      if (progress >= 100) {
        clearInterval(interval)
        setIsProcessing(false)
        // Here you would normally handle the actual processing result
        // For now, we'll just simulate completion
      }
    }, 200)
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`
  }

  const currentVideo = recordedVideo || uploadedVideo

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Road Scene Video Capture</CardTitle>
          <CardDescription>Record or upload videos of road scenes for analysis</CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              {!currentVideo ? (
                <>
                  <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                    <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />

                    {isRecording && (
                      <div className="absolute top-4 right-4 bg-red-500 text-white px-2 py-1 rounded-md flex items-center">
                        <div className="w-3 h-3 rounded-full bg-white mr-2 animate-pulse"></div>
                        {formatTime(recordingDuration)}
                      </div>
                    )}
                  </div>

                  <div className="flex justify-center gap-4">
                    {!isRecording ? (
                      <Button onClick={startRecording}>
                        <Video className="mr-2 h-4 w-4" />
                        Start Recording
                      </Button>
                    ) : (
                      <Button variant="destructive" onClick={stopRecording}>
                        <Square className="mr-2 h-4 w-4" />
                        Stop Recording
                      </Button>
                    )}

                    <input
                      type="file"
                      accept="video/*"
                      className="hidden"
                      ref={fileInputRef}
                      onChange={handleFileUpload}
                    />
                    <Button variant="outline" onClick={() => fileInputRef.current?.click()}>
                      <Upload className="mr-2 h-4 w-4" />
                      Upload Video
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                    {recordedVideo ? (
                      <video
                        ref={recordedVideoRef}
                        className="w-full h-full object-contain"
                        onTimeUpdate={handleTimeUpdate}
                        onEnded={() => setIsPlaying(false)}
                      />
                    ) : (
                      <video
                        ref={uploadedVideoRef}
                        className="w-full h-full object-contain"
                        onTimeUpdate={handleTimeUpdate}
                        onEnded={() => setIsPlaying(false)}
                      />
                    )}
                  </div>

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

                  <div className="flex justify-center gap-4">
                    <Button variant="outline" onClick={resetRecording}>
                      <RefreshCw className="mr-2 h-4 w-4" />
                      New Video
                    </Button>

                    <Button onClick={downloadVideo}>
                      <Download className="mr-2 h-4 w-4" />
                      Download
                    </Button>

                    <Button onClick={processVideo} disabled={isProcessing}>
                      <Video className="mr-2 h-4 w-4" />
                      {isProcessing ? "Processing..." : "Analyze Video"}
                    </Button>
                  </div>
                </>
              )}
            </div>

            <div className="space-y-4">
              {isProcessing ? (
                <div className="space-y-4">
                  <h3 className="font-medium">Processing Video</h3>
                  <Progress value={processingProgress} className="h-2" />
                  <p className="text-sm text-muted-foreground">
                    Analyzing road objects in video... {processingProgress}%
                  </p>
                </div>
              ) : (
                <div className="bg-muted rounded-lg p-6 h-full flex items-center justify-center">
                  <div className="text-center">
                    <h3 className="font-medium mb-2">Video Analysis</h3>
                    <p className="text-sm text-muted-foreground">
                      {currentVideo
                        ? "Click 'Analyze Video' to detect road objects and generate insights"
                        : "Record or upload a video to begin analysis"}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Tips for Road Video Capture</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc pl-5 space-y-2">
            <li>Mount your camera securely to avoid shaky footage</li>
            <li>Record at least 10-15 seconds of footage for accurate analysis</li>
            <li>Try to capture natural traffic flow and road conditions</li>
            <li>For dashboard cameras, ensure the windshield is clean</li>
            <li>Record in good lighting conditions when possible</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
