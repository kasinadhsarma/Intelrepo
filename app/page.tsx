"use client"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ImageCapture } from "@/components/image-capture"
import { VideoCapture } from "@/components/video-capture"
import { LiveCapture } from "@/components/live-capture"

export default function Home() {
  return (
    <main className="container mx-auto py-10 px-4">
      <h1 className="text-3xl font-bold mb-2 text-center">Road Object Detection</h1>
      <p className="text-center text-muted-foreground mb-6">Detect and analyze objects in road scenes</p>

      <Tabs defaultValue="image" className="w-full max-w-4xl mx-auto">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="image">Image Detection</TabsTrigger>
          <TabsTrigger value="video">Video Detection</TabsTrigger>
          <TabsTrigger value="live">Live Detection</TabsTrigger>
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

