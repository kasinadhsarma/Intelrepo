/**
 * API client for interacting with the FastAPI backend
 */

const API_BASE_URL = "http://localhost:8000"

export async function detectObjects(imageBlob: Blob) {
  const formData = new FormData()
  formData.append("file", imageBlob, "image.jpg")

  const response = await fetch(`${API_BASE_URL}/detect_objects/`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }

  return response.json()
}

export async function segmentImage(imageBlob: Blob) {
  const formData = new FormData()
  formData.append("file", imageBlob, "image.jpg")

  const response = await fetch(`${API_BASE_URL}/segment_image/`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }

  return response.json()
}

export async function processVideo(videoBlob: Blob) {
  const formData = new FormData()
  formData.append("file", videoBlob, "video.webm")

  const response = await fetch(`${API_BASE_URL}/process_video/`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }

  return response.json()
}

export async function healthCheck() {
  const response = await fetch(`${API_BASE_URL}/health`)

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }

  return response.json()
}

export async function getRoadSafetyInfo(imageBlob: Blob) {
  const formData = new FormData()
  formData.append("file", imageBlob, "image.jpg")

  const response = await fetch(`${API_BASE_URL}/road_safety/`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }

  return response.json()
}

