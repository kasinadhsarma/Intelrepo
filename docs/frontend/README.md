# Frontend Documentation

## Overview

The frontend is built with Next.js 15.2 and provides a modern, responsive interface for road object detection tasks.

## Components

### Core Components

1. **ImageCapture**
- Handles single image uploads
- Supports drag-and-drop functionality
- Displays detection results with bounding boxes
- Shows AI-generated scene descriptions

2. **VideoCapture**
- Processes video files
- Shows progress tracking
- Performs frame-by-frame analysis
- Provides exportable results

3. **LiveCapture**
- Real-time camera feed processing
- FPS counter display
- Live object detection visualization
- Adjustable detection settings

### UI Components

Built using a comprehensive UI kit with Radix UI primitives:
- Cards for structured content display
- Dialogs for user interactions
- Tooltips for enhanced usability
- Progress indicators for operations
- Alert components for notifications
- Custom buttons and form inputs

## State Management

- Context-based state management
- Efficient render optimization
- Real-time updates handling

## API Integration

The frontend integrates with the backend API through:
- Dedicated API client (`api-client.ts`)
- File upload handling
- Streaming response processing
- Error state management

## Performance Optimizations

1. **Image Processing**
- Client-side image compression
- Efficient canvas rendering
- WebGL acceleration when available

2. **Video Handling**
- Chunked upload support
- Stream processing
- Frame rate optimization

3. **Real-time Detection**
- Frame skipping for performance
- Resolution adaptation
- WebWorker processing

## Directory Structure

```
components/
├── image-capture.tsx      # Image processing component
├── live-capture.tsx       # Real-time camera component
├── video-capture.tsx      # Video processing component
├── theme-provider.tsx     # Theme management
└── ui/                    # Reusable UI components
    ├── button.tsx
    ├── card.tsx
    └── ...

lib/
├── api-client.ts         # API integration
└── utils.ts              # Utility functions
```

## Development Guide

### Adding New Features

1. Create new components in `components/`
2. Update API client if needed
3. Add tests for new functionality
4. Update documentation

### Code Style

- Follow TypeScript best practices
- Use functional components
- Implement proper error handling
- Add JSDoc comments for complex logic

### Testing

Run tests using:
```bash
npm test
# or
pnpm test
```

## Troubleshooting

Common frontend issues and solutions:

1. **Performance Issues**
- Check browser console for warnings
- Monitor memory usage
- Verify WebGL support

2. **API Connection**
- Verify API endpoint configuration
- Check CORS settings
- Monitor network requests

3. **Camera Access**
- Ensure proper permissions
- Check SSL configuration
- Verify browser compatibility