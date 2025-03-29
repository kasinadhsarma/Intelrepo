import { NextResponse } from 'next/server'
import { prisma } from '@/lib/db'

export async function POST(request: Request) {
  try {
    const data = await request.json()
    
    // Store detection results in database
    const result = await prisma.detection.create({
      data: {
        type: 'video',
        objects: data.objects,
        analysis: {
          roadConditions: data.roadConditions,
          trafficDensity: data.trafficDensity,
          avgSpeed: data.avgSpeed,
          safetyScore: data.safetyScore,
          riskAreas: data.riskAreas
        }
      }
    })

    return NextResponse.json({ success: true, id: result.id })
  } catch (error) {
    console.error('Error storing video results:', error)
    return NextResponse.json({ error: 'Failed to store results' }, { status: 500 })
  }
}