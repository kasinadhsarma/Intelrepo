"use client"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ImageCapture } from "@/components/image-capture"
import { VideoCapture } from "@/components/video-capture"
import { LiveCapture } from "@/components/live-capture"

export default function Home() {
  return (
    <main className="container mx-auto py-10 px-4">
      <h1 className="text-3xl font-bold mb-2 text-center">DeepAction Experimental</h1>
      <p className="text-center text-muted-foreground mb-6">Road Object Detection & Analysis</p>

      <Tabs defaultValue="image" className="w-full max-w-4xl mx-auto">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="image">Image Capture</TabsTrigger>
          <TabsTrigger value="video">Video Capture</TabsTrigger>
          <TabsTrigger value="live">Live Capture</TabsTrigger>
        </TabsList>
        <TabsContent value="image" className="mt-6">
          <ImageCapture />
        </TabsContent>
        <TabsContent value="video" className="mt-6">
          <VideoCapture />
        </TabsContent>
        <TabsContent value="live" className="mt-6">
          <LiveCapture />
        </TabsContent>
      </Tabs>
    </main>
  )
}

