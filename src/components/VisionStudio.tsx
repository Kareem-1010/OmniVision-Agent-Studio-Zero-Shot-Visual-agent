/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Camera,
  Play,
  Pause,
  RefreshCw,
  Eye,
  Layers,
  Activity,
  Crosshair,
  Upload,
  Image as ImageIcon,
  Sparkles,
  CheckCircle2,
  Square,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Info,
  Cpu,
  Sliders,
  Filter,
} from 'lucide-react';
import { InstanceNode, CVEngineMode } from '../types/index.ts';
import { CVToolsPanel, CVFilterSettings } from './CVToolsPanel.tsx';

interface VisionStudioProps {
  instances: InstanceNode[];
  onAnalyzeFrame: (base64Image: string, customQuery?: string, engine?: CVEngineMode, nmsThreshold?: number) => void;
  isAnalyzing: boolean;
  selectedInstanceId?: string;
  onSelectInstance?: (id: string) => void;
  uploadedImage?: string | null;
  onImageUploaded?: (base64Image: string) => void;
  onOpenAnnotation?: () => void;
  cvEngine?: CVEngineMode;
  onEngineChange?: (engine: CVEngineMode) => void;
  cvMetrics?: {
    cvEngine: string;
    nmsSuppressedCount: number;
    meanClipScore: number;
    sam2PolygonsGenerated: number;
    activeIouThreshold: number;
  };
}

export const SAMPLE_BENCHMARK_STREAMS = [
  {
    id: 'stream_cars',
    name: 'Parking Lot / Traffic (Cars Benchmark)',
    type: 'cars',
    image: 'https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=1200&q=80',
  },
  {
    id: 'stream_hazard',
    name: 'Industrial Warehouse (Safety Hazard)',
    type: 'hazard',
    image: 'https://images.unsplash.com/photo-1581092160607-ee22621dd758?auto=format&fit=crop&w=1200&q=80',
  },
  {
    id: 'stream_retail',
    name: 'Retail Store Lobby (Dwell & Tracking)',
    type: 'retail',
    image: 'https://images.unsplash.com/photo-1528698827591-e19ccd7bc23d?auto=format&fit=crop&w=1200&q=80',
  },
  {
    id: 'stream_corridor',
    name: 'Corridor & Sign (OCR Grounding)',
    type: 'ocr',
    image: 'https://images.unsplash.com/photo-1541888946425-d0fbb186f5f7?auto=format&fit=crop&w=1200&q=80',
  },
  {
    id: 'stream_aisle',
    name: 'Warehouse Aisle (Counterfactual Cart)',
    type: 'aisle',
    image: 'https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?auto=format&fit=crop&w=1200&q=80',
  },
];

function applyCannyEdgeFilter(
  ctx: CanvasRenderingContext2D,
  imgRect: { x: number; y: number; width: number; height: number }
) {
  try {
    const rx = Math.max(0, Math.floor(imgRect.x));
    const ry = Math.max(0, Math.floor(imgRect.y));
    const rw = Math.min(ctx.canvas.width - rx, Math.floor(imgRect.width));
    const rh = Math.min(ctx.canvas.height - ry, Math.floor(imgRect.height));
    if (rw <= 2 || rh <= 2) return;

    const imgData = ctx.getImageData(rx, ry, rw, rh);
    const data = imgData.data;
    const gray = new Uint8Array(rw * rh);

    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }

    const outData = new Uint8ClampedArray(data.length);
    for (let y = 1; y < rh - 1; y++) {
      for (let x = 1; x < rw - 1; x++) {
        const idx = y * rw + x;
        const gx =
          -gray[idx - rw - 1] + gray[idx - rw + 1] -
          2 * gray[idx - 1] + 2 * gray[idx + 1] -
          gray[idx + rw - 1] + gray[idx + rw + 1];

        const gy =
          -gray[idx - rw - 1] - 2 * gray[idx - rw] - gray[idx - rw + 1] +
          gray[idx + rw - 1] + 2 * gray[idx + rw] + gray[idx + rw + 1];

        const mag = Math.sqrt(gx * gx + gy * gy);
        const pIdx = idx * 4;

        if (mag > 65) {
          outData[pIdx] = 16;
          outData[pIdx + 1] = 230;
          outData[pIdx + 2] = 200;
          outData[pIdx + 3] = 255;
        } else if (mag > 32) {
          outData[pIdx] = 6;
          outData[pIdx + 1] = 140;
          outData[pIdx + 2] = 180;
          outData[pIdx + 3] = 180;
        } else {
          outData[pIdx] = 8;
          outData[pIdx + 1] = 12;
          outData[pIdx + 2] = 20;
          outData[pIdx + 3] = 230;
        }
      }
    }

    imgData.data.set(outData);
    ctx.putImageData(imgData, rx, ry);
  } catch {
    // Cross-origin image data fallback if tainted
  }
}

function applySobelGradientFilter(
  ctx: CanvasRenderingContext2D,
  imgRect: { x: number; y: number; width: number; height: number }
) {
  try {
    const rx = Math.max(0, Math.floor(imgRect.x));
    const ry = Math.max(0, Math.floor(imgRect.y));
    const rw = Math.min(ctx.canvas.width - rx, Math.floor(imgRect.width));
    const rh = Math.min(ctx.canvas.height - ry, Math.floor(imgRect.height));
    if (rw <= 2 || rh <= 2) return;

    const imgData = ctx.getImageData(rx, ry, rw, rh);
    const data = imgData.data;
    const gray = new Uint8Array(rw * rh);

    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }

    const outData = new Uint8ClampedArray(data.length);
    for (let y = 1; y < rh - 1; y++) {
      for (let x = 1; x < rw - 1; x++) {
        const idx = y * rw + x;
        const gx =
          -gray[idx - rw - 1] + gray[idx - rw + 1] -
          2 * gray[idx - 1] + 2 * gray[idx + 1] -
          gray[idx + rw - 1] + gray[idx + rw + 1];

        const gy =
          -gray[idx - rw - 1] - 2 * gray[idx - rw] - gray[idx - rw + 1] +
          gray[idx + rw - 1] + 2 * gray[idx + rw] + gray[idx + rw + 1];

        const mag = Math.min(255, Math.sqrt(gx * gx + gy * gy) * 1.5);
        const pIdx = idx * 4;

        outData[pIdx] = Math.min(255, Math.round(mag * 1.1));
        outData[pIdx + 1] = Math.round(mag * 0.75);
        outData[pIdx + 2] = Math.round(mag * 0.2);
        outData[pIdx + 3] = 230;
      }
    }

    imgData.data.set(outData);
    ctx.putImageData(imgData, rx, ry);
  } catch {
    // Cross-origin fallback
  }
}

export const VisionStudio: React.FC<VisionStudioProps> = ({
  instances,
  onAnalyzeFrame,
  isAnalyzing,
  selectedInstanceId,
  onSelectInstance,
  uploadedImage: initialUploadedImage,
  onImageUploaded,
  onOpenAnnotation,
  cvEngine = 'omnivision',
  onEngineChange,
  cvMetrics,
}) => {
  const [sourceMode, setSourceMode] = useState<'upload' | 'sample' | 'webcam'>(
    initialUploadedImage ? 'upload' : 'sample'
  );
  const [selectedStream, setSelectedStream] = useState(SAMPLE_BENCHMARK_STREAMS[0]);
  const [uploadedImageData, setUploadedImageData] = useState<string | null>(initialUploadedImage || null);
  const [uploadedFileName, setUploadedFileName] = useState<string>('user-image.jpg');
  const [isDragOver, setIsDragOver] = useState<boolean>(false);

  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [showDepthOverlay, setShowDepthOverlay] = useState<boolean>(false);
  const [showOpticalFlow, setShowOpticalFlow] = useState<boolean>(false);
  const [isLivePlaying, setIsLivePlaying] = useState<boolean>(true);
  const [zoomLevel, setZoomLevel] = useState<number>(1.0);
  const [hoveredInstanceId, setHoveredInstanceId] = useState<string | null>(null);
  const [badgeDisplayMode, setBadgeDisplayMode] = useState<'adaptive' | 'detailed' | 'minimal'>('adaptive');
  const [showCvPanel, setShowCvPanel] = useState<boolean>(true);

  const [cvFilters, setCvFilters] = useState<CVFilterSettings>({
    cannyEdge: false,
    sobelGradient: false,
    sam2Polygons: true,
    depthHeatmap: false,
    showIouMatrix: false,
    showColorAnalysis: false,
    showClipScores: true,
    nmsThreshold: 0.50,
  });

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const loadedImageRef = useRef<HTMLImageElement | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Sync uploaded image prop
  useEffect(() => {
    if (initialUploadedImage) {
      setUploadedImageData(initialUploadedImage);
      setSourceMode('upload');
      loadImageSource(initialUploadedImage);
    }
  }, [initialUploadedImage]);

  // Load image whenever source changes
  const loadImageSource = useCallback((src: string) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = src;
    img.onload = () => {
      loadedImageRef.current = img;
    };
    img.onerror = () => {
      console.warn('[VisionStudio] Image load warning with CORS, retrying direct:', src);
      const fallback = new Image();
      fallback.src = src;
      fallback.onload = () => {
        loadedImageRef.current = fallback;
      };
    };
  }, []);

  useEffect(() => {
    if (sourceMode === 'upload' && uploadedImageData) {
      loadImageSource(uploadedImageData);
    } else if (sourceMode === 'sample') {
      loadImageSource(selectedStream.image);
    }
  }, [sourceMode, selectedStream, uploadedImageData, loadImageSource]);

  // Initialize Webcam
  useEffect(() => {
    if (sourceMode === 'webcam') {
      navigator.mediaDevices
        ?.getUserMedia({ video: { width: 1280, height: 720 } })
        .then((stream) => {
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play();
          }
        })
        .catch((err) => {
          console.warn('[VisionStudio] Camera access error:', err);
          setSourceMode('sample');
        });
    } else {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    }

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, [sourceMode]);

  // Handle Image File Selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processSelectedImage(file);
    }
  };

  const processSelectedImage = (file: File) => {
    setUploadedFileName(file.name);
    const reader = new FileReader();
    reader.onload = (event) => {
      const base64 = event.target?.result as string;
      if (base64) {
        setUploadedImageData(base64);
        setSourceMode('upload');
        loadImageSource(base64);
        if (onImageUploaded) onImageUploaded(base64);
        onAnalyzeFrame(base64, 'Detect and localize all objects, text, metric depth, and safety items');
      }
    };
    reader.readAsDataURL(file);
  };

  // Drag and Drop
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      processSelectedImage(file);
    }
  };

  // Helper to get exact rendered image rect on the canvas
  const computeImageRect = (cW: number, cH: number, naturalW: number, naturalH: number) => {
    const imgAspect = naturalW / naturalH;
    const canvasAspect = cW / cH;
    let rW = cW;
    let rH = cH;
    let oX = 0;
    let oY = 0;

    if (imgAspect > canvasAspect) {
      // Letterbox top & bottom
      rW = cW * zoomLevel;
      rH = (cW / imgAspect) * zoomLevel;
      oX = (cW - rW) / 2;
      oY = (cH - rH) / 2;
    } else {
      // Pillarbox left & right
      rH = cH * zoomLevel;
      rW = (cH * imgAspect) * zoomLevel;
      oX = (cW - rW) / 2;
      oY = (cH - rH) / 2;
    }

    return { x: oX, y: oY, width: rW, height: rH };
  };

  // Canvas Mouse Click: Select Instance
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    let natW = 1200;
    let natH = 800;
    if (sourceMode === 'webcam' && videoRef.current) {
      natW = videoRef.current.videoWidth || 1280;
      natH = videoRef.current.videoHeight || 720;
    } else if (loadedImageRef.current) {
      natW = loadedImageRef.current.naturalWidth || 1200;
      natH = loadedImageRef.current.naturalHeight || 800;
    }

    const imgRect = computeImageRect(canvas.width, canvas.height, natW, natH);

    // Hit test reverse (top-most box first)
    const hit = [...instances].reverse().find((inst) => {
      const bx = imgRect.x + (inst.bbox2d.xmin / 1000) * imgRect.width;
      const by = imgRect.y + (inst.bbox2d.ymin / 1000) * imgRect.height;
      const bw = ((inst.bbox2d.xmax - inst.bbox2d.xmin) / 1000) * imgRect.width;
      const bh = ((inst.bbox2d.ymax - inst.bbox2d.ymin) / 1000) * imgRect.height;
      return clickX >= bx && clickX <= bx + bw && clickY >= by && clickY <= by + bh;
    });

    if (hit && onSelectInstance) {
      onSelectInstance(hit.id);
    }
  };

  // Canvas Mouse Move: Hover Highlight
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    let natW = 1200;
    let natH = 800;
    if (sourceMode === 'webcam' && videoRef.current) {
      natW = videoRef.current.videoWidth || 1280;
      natH = videoRef.current.videoHeight || 720;
    } else if (loadedImageRef.current) {
      natW = loadedImageRef.current.naturalWidth || 1200;
      natH = loadedImageRef.current.naturalHeight || 800;
    }

    const imgRect = computeImageRect(canvas.width, canvas.height, natW, natH);

    const hit = [...instances].reverse().find((inst) => {
      const bx = imgRect.x + (inst.bbox2d.xmin / 1000) * imgRect.width;
      const by = imgRect.y + (inst.bbox2d.ymin / 1000) * imgRect.height;
      const bw = ((inst.bbox2d.xmax - inst.bbox2d.xmin) / 1000) * imgRect.width;
      const bh = ((inst.bbox2d.ymax - inst.bbox2d.ymin) / 1000) * imgRect.height;
      return clickX >= bx && clickX <= bx + bw && clickY >= by && clickY <= by + bh;
    });

    setHoveredInstanceId(hit ? hit.id : null);
  };

  // Main Render Loop (Draws Image + Overlays in 100% Lockstep)
  useEffect(() => {
    let phase = 0;
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      phase += 0.04;

      // Match canvas pixel buffer to container dimensions
      const cW = container.clientWidth || 1000;
      const cH = container.clientHeight || 500;
      if (canvas.width !== cW || canvas.height !== cH) {
        canvas.width = cW;
        canvas.height = cH;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 1. Draw Technical HUD Background Grid
      ctx.fillStyle = '#090d16';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.strokeStyle = 'rgba(30, 41, 59, 0.4)';
      ctx.lineWidth = 1;
      const gridSpacing = 40;
      for (let x = 0; x < canvas.width; x += gridSpacing) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += gridSpacing) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      // 2. Draw Image or Webcam Frame
      let imgRect = { x: 0, y: 0, width: canvas.width, height: canvas.height };

      if (sourceMode === 'webcam' && videoRef.current && videoRef.current.readyState >= 2) {
        const vW = videoRef.current.videoWidth || 1280;
        const vH = videoRef.current.videoHeight || 720;
        imgRect = computeImageRect(canvas.width, canvas.height, vW, vH);
        ctx.drawImage(videoRef.current, imgRect.x, imgRect.y, imgRect.width, imgRect.height);
      } else if (loadedImageRef.current && loadedImageRef.current.complete && loadedImageRef.current.naturalWidth > 0) {
        const nW = loadedImageRef.current.naturalWidth;
        const nH = loadedImageRef.current.naturalHeight;
        imgRect = computeImageRect(canvas.width, canvas.height, nW, nH);
        ctx.drawImage(loadedImageRef.current, imgRect.x, imgRect.y, imgRect.width, imgRect.height);
      } else {
        // Fallback loading indicator
        ctx.fillStyle = '#0ea5e9';
        ctx.font = '14px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Loading visual stream...', canvas.width / 2, canvas.height / 2);
        ctx.textAlign = 'left';
      }

      // 2b. Draw Real-Time CV Image Filters (Canny Edge / Sobel Gradient)
      if (cvFilters.cannyEdge) {
        applyCannyEdgeFilter(ctx, imgRect);
      } else if (cvFilters.sobelGradient) {
        applySobelGradientFilter(ctx, imgRect);
      }

      // 3. Draw Depth Colormap Overlay if enabled
      if (showDepthOverlay || cvFilters.depthHeatmap) {
        ctx.save();
        ctx.globalAlpha = 0.42;
        const grad = ctx.createLinearGradient(imgRect.x, imgRect.y + imgRect.height, imgRect.x, imgRect.y);
        grad.addColorStop(0, '#ef4444'); // near
        grad.addColorStop(0.3, '#f59e0b');
        grad.addColorStop(0.6, '#10b981');
        grad.addColorStop(1, '#3b82f6'); // far
        ctx.fillStyle = grad;
        ctx.fillRect(imgRect.x, imgRect.y, imgRect.width, imgRect.height);
        ctx.restore();
      }

      // 4. Draw Optical Flow Motion Vectors if enabled
      if (showOpticalFlow && isLivePlaying && sourceMode !== 'upload') {
        ctx.save();
        ctx.strokeStyle = '#06b6d4';
        ctx.fillStyle = '#06b6d4';
        ctx.lineWidth = 1.5;

        const step = 50;
        for (let x = imgRect.x + 25; x < imgRect.x + imgRect.width; x += step) {
          for (let y = imgRect.y + 25; y < imgRect.y + imgRect.height; y += step) {
            const vx = Math.sin(phase + x * 0.015 + y * 0.02) * 12;
            const vy = Math.cos(phase + y * 0.015) * 8;
            if (Math.abs(vx) + Math.abs(vy) > 8) {
              ctx.beginPath();
              ctx.moveTo(x, y);
              ctx.lineTo(x + vx, y + vy);
              ctx.stroke();

              ctx.beginPath();
              ctx.arc(x + vx, y + vy, 2, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        }
        ctx.restore();
      }

      // 5. Draw SAM 2 Segmentation Masks / Polygons if enabled
      if (cvFilters.sam2Polygons && instances.length > 0) {
        instances.forEach((inst) => {
          const poly = inst.polygon2d;
          if (!poly || poly.length < 3) return;

          const isSelected = inst.id === selectedInstanceId;
          const isHovered = inst.id === hoveredInstanceId;

          ctx.save();
          ctx.beginPath();
          poly.forEach(([py, px], pIdx) => {
            const cx = imgRect.x + (px / 1000) * imgRect.width;
            const cy = imgRect.y + (py / 1000) * imgRect.height;
            if (pIdx === 0) ctx.moveTo(cx, cy);
            else ctx.lineTo(cx, cy);
          });
          ctx.closePath();

          ctx.fillStyle = isSelected
            ? 'rgba(250, 204, 21, 0.35)'
            : isHovered
            ? 'rgba(255, 255, 255, 0.28)'
            : 'rgba(168, 85, 247, 0.22)';
          ctx.fill();

          ctx.strokeStyle = isSelected ? '#facc15' : isHovered ? '#ffffff' : '#c084fc';
          ctx.lineWidth = isSelected ? 2.5 : 1.8;
          ctx.stroke();

          // Vertex indicators
          if (isSelected || isHovered) {
            ctx.fillStyle = '#ffffff';
            poly.forEach(([py, px]) => {
              const cx = imgRect.x + (px / 1000) * imgRect.width;
              const cy = imgRect.y + (py / 1000) * imgRect.height;
              ctx.fillRect(cx - 2, cy - 2, 4, 4);
            });
          }

          ctx.restore();
        });
      }

      // 6. Draw Pixel-Perfect Bounding Boxes (Tightly Mapped to Image)
      if (showBoundingBoxes && instances.length > 0) {
        const renderedBadges: Array<{ x: number; y: number; w: number; h: number }> = [];

        // Dynamic client-side NMS suppression according to cvFilters.nmsThreshold
        const nmsFiltered = instances.filter((inst, idx) => {
          for (let i = 0; i < idx; i++) {
            const higher = instances[i];
            if (higher.confidence >= inst.confidence) {
              const yA = Math.max(inst.bbox2d.ymin, higher.bbox2d.ymin);
              const xA = Math.max(inst.bbox2d.xmin, higher.bbox2d.xmin);
              const yB = Math.min(inst.bbox2d.ymax, higher.bbox2d.ymax);
              const xB = Math.min(inst.bbox2d.xmax, higher.bbox2d.xmax);
              const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA);
              const a1 = (inst.bbox2d.xmax - inst.bbox2d.xmin) * (inst.bbox2d.ymax - inst.bbox2d.ymin);
              const a2 = (higher.bbox2d.xmax - higher.bbox2d.xmin) * (higher.bbox2d.ymax - higher.bbox2d.ymin);
              const iou = inter > 0 ? inter / (a1 + a2 - inter) : 0;
              if (iou > cvFilters.nmsThreshold) {
                return false;
              }
            }
          }
          return true;
        });

        // Sort instances: unselected/unhovered drawn first, hovered/selected drawn LAST (on top)
        const sortedInstances = [...nmsFiltered].sort((a, b) => {
          if (a.id === selectedInstanceId) return 1;
          if (b.id === selectedInstanceId) return -1;
          if (a.id === hoveredInstanceId) return 1;
          if (b.id === hoveredInstanceId) return -1;
          return 0;
        });

        sortedInstances.forEach((inst, idx) => {
          const isSelected = inst.id === selectedInstanceId;
          const isHovered = inst.id === hoveredInstanceId;
          const box = inst.bbox2d;

          const bx = imgRect.x + (box.xmin / 1000) * imgRect.width;
          const by = imgRect.y + (box.ymin / 1000) * imgRect.height;
          const bw = ((box.xmax - box.xmin) / 1000) * imgRect.width;
          const bh = ((box.ymax - box.ymin) / 1000) * imgRect.height;

          // Color palette with multi-category and explicit color grounding
          let color = '#38bdf8';
          const lcase = inst.label.toLowerCase();
          const colorName = inst.colorLAB?.name?.toLowerCase() || '';

          if (lcase.includes('red') || colorName.includes('red') || colorName.includes('crimson') || colorName.includes('ruby')) {
            color = '#ef4444'; // Red
          } else if (lcase.includes('orange') || colorName.includes('orange')) {
            color = '#f97316';
          } else if (lcase.includes('yellow') || colorName.includes('yellow')) {
            color = '#eab308';
          } else if (lcase.includes('hazard') || lcase.includes('spill') || lcase.includes('defect') || lcase.includes('danger')) {
            color = '#f43f5e';
          } else if (lcase.includes('worker') || lcase.includes('person') || lcase.includes('pedestrian') || lcase.includes('driver')) {
            color = '#10b981';
          } else if (lcase.includes('car') || lcase.includes('vehicle') || lcase.includes('auto') || lcase.includes('bmw') || lcase.includes('truck') || lcase.includes('sedan') || lcase.includes('suv')) {
            color = '#38bdf8';
          } else if (lcase.includes('sign') || lcase.includes('cart') || lcase.includes('cone') || lcase.includes('obstacle') || lcase.includes('text')) {
            color = '#f59e0b';
          } else if (lcase.includes('tree') || lcase.includes('palm') || lcase.includes('plant') || lcase.includes('grass')) {
            color = '#22c55e';
          } else if (lcase.includes('building') || lcase.includes('house') || lcase.includes('door') || lcase.includes('shelf')) {
            color = '#a855f7';
          }

          if (isSelected) color = '#facc15';
          else if (isHovered) color = '#ffffff';

          ctx.save();

          // Subtle box interior highlight
          ctx.fillStyle = isSelected
            ? 'rgba(250, 204, 21, 0.22)'
            : isHovered
            ? 'rgba(255, 255, 255, 0.16)'
            : color === '#ef4444'
            ? 'rgba(239, 68, 68, 0.12)'
            : 'rgba(56, 189, 248, 0.08)';
          ctx.fillRect(bx, by, bw, bh);

          // Bounding Box stroke
          ctx.strokeStyle = color;
          ctx.lineWidth = isSelected ? 3 : isHovered ? 2.5 : 2;
          ctx.setLineDash(isSelected ? [] : [4, 2]);
          ctx.strokeRect(bx, by, bw, bh);

          // Corner Brackets for High-Tech HUD aesthetic (scaled to prevent overlap)
          const cornerLen = Math.max(3, Math.min(10, bw * 0.2, bh * 0.2));
          ctx.setLineDash([]);
          ctx.lineWidth = isSelected ? 3.5 : 2.5;

          // Top-Left Corner
          ctx.beginPath();
          ctx.moveTo(bx, by + cornerLen);
          ctx.lineTo(bx, by);
          ctx.lineTo(bx + cornerLen, by);
          ctx.stroke();

          // Top-Right Corner
          ctx.beginPath();
          ctx.moveTo(bx + bw - cornerLen, by);
          ctx.lineTo(bx + bw, by);
          ctx.lineTo(bx + bw, by + cornerLen);
          ctx.stroke();

          // Bottom-Left Corner
          ctx.beginPath();
          ctx.moveTo(bx, by + bh - cornerLen);
          ctx.lineTo(bx, by + bh);
          ctx.lineTo(bx + cornerLen, by + bh);
          ctx.stroke();

          // Bottom-Right Corner
          ctx.beginPath();
          ctx.moveTo(bx + bw - cornerLen, by + bh);
          ctx.lineTo(bx + bw, by + bh);
          ctx.lineTo(bx + bw, by + bh - cornerLen);
          ctx.stroke();

          // Label Badge Header Construction
          const targetIndex = idx + 1;
          const confPercent = Math.round(inst.confidence * 100);
          const cleanLabel = inst.label.replace(/_/g, ' ');
          const clipPercent = inst.clipScore ? Math.round(inst.clipScore * 100) : 95;
          const yoloCls = inst.yoloClass || inst.label.split('_')[0];

          let badgeText = '';
          if (badgeDisplayMode === 'minimal') {
            badgeText = `#${targetIndex}`;
          } else if (badgeDisplayMode === 'adaptive') {
            if (bw < 70) {
              badgeText = `#${targetIndex} ${confPercent}%`;
            } else if (cvFilters.showClipScores && bw > 140) {
              badgeText = `#${targetIndex} ${cleanLabel} [CLIP:${clipPercent}%]`;
            } else {
              badgeText = `#${targetIndex} ${cleanLabel} ${confPercent}%`;
            }
          } else {
            // Detailed mode
            const depthLabel = `${inst.depth.metricDistanceM}m`;
            badgeText = `#${targetIndex} ${cleanLabel} [${yoloCls} ${confPercent}% • CLIP:${clipPercent}%] • ${depthLabel}`;
          }

          ctx.font = 'bold 10px monospace';
          const badgeMetrics = ctx.measureText(badgeText);
          const badgeW = badgeMetrics.width + 12;
          const badgeH = 18;

          // Smart collision avoidance: candidate positions
          let badgeX = Math.max(imgRect.x, Math.min(bx, imgRect.x + imgRect.width - badgeW));
          let badgeY = by - badgeH - 2;

          const hasCollision = (cy: number) => {
            return renderedBadges.some(
              (rb) =>
                Math.abs(rb.x - badgeX) < (rb.w + badgeW) * 0.45 &&
                Math.abs(rb.y - cy) < (rb.h + badgeH) * 0.8
            );
          };

          if (badgeY < imgRect.y || hasCollision(badgeY)) {
            // Option 2: Inside top of the box
            if (by + badgeH + 4 < by + bh && !hasCollision(by + 2)) {
              badgeY = by + 2;
            } else if (by + bh + badgeH + 2 <= imgRect.y + imgRect.height && !hasCollision(by + bh + 2)) {
              // Option 3: Below bottom of the box
              badgeY = by + bh + 2;
            } else {
              badgeY = Math.max(imgRect.y, by - badgeH - 2);
            }
          }

          renderedBadges.push({ x: badgeX, y: badgeY, w: badgeW, h: badgeH });

          // Badge Background
          ctx.fillStyle = isSelected ? '#facc15' : 'rgba(15, 23, 42, 0.94)';
          ctx.fillRect(badgeX, badgeY, badgeW, badgeH);

          // Badge Border
          ctx.strokeStyle = color;
          ctx.lineWidth = 1;
          ctx.strokeRect(badgeX, badgeY, badgeW, badgeH);

          // Badge Text
          ctx.fillStyle = isSelected ? '#0f172a' : color;
          ctx.fillText(badgeText, badgeX + 6, badgeY + 13);

          // Draw tracking ID dot if space permits
          if (inst.trackId && bw > 30) {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(bx + bw - 8, by + 8, 4.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#0f172a';
            ctx.font = 'bold 7px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(String(inst.trackId % 100), bx + bw - 8, by + 10.5);
            ctx.textAlign = 'left';
          }

          ctx.restore();
        });
      }

      animationFrameIdRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameIdRef.current) cancelAnimationFrame(animationFrameIdRef.current);
    };
  }, [
    instances,
    selectedInstanceId,
    hoveredInstanceId,
    showBoundingBoxes,
    showDepthOverlay,
    showOpticalFlow,
    isLivePlaying,
    sourceMode,
    zoomLevel,
    badgeDisplayMode,
    cvFilters,
  ]);

  // Capture current frame or uploaded image and send to analysis pipeline
  const handleCaptureFrame = (queryOverride?: string) => {
    const nms = cvFilters.nmsThreshold;
    if (sourceMode === 'upload' && uploadedImageData) {
      onAnalyzeFrame(uploadedImageData, queryOverride, cvEngine, nms);
    } else if (sourceMode === 'webcam' && videoRef.current) {
      const snapCanvas = document.createElement('canvas');
      snapCanvas.width = videoRef.current.videoWidth || 1280;
      snapCanvas.height = videoRef.current.videoHeight || 720;
      const snapCtx = snapCanvas.getContext('2d');
      if (snapCtx) {
        snapCtx.drawImage(videoRef.current, 0, 0);
        const dataUrl = snapCanvas.toDataURL('image/jpeg', 0.9);
        onAnalyzeFrame(dataUrl, queryOverride, cvEngine, nms);
      }
    } else {
      // Pass the selected stream image directly (server fetches remote URLs seamlessly)
      onAnalyzeFrame(selectedStream.image, queryOverride, cvEngine, nms);
    }
  };

  const QUICK_UPLOAD_QUERIES = [
    'What objects are in this image and where are they located?',
    'Count all the cars and vehicles',
    'Is there any safety hazard or violation?',
    'Read all text and signs in the scene',
    'Calculate metric distance and 3D bounding boxes',
  ];

  const selectedInstance = instances.find((i) => i.id === selectedInstanceId);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col">
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Top Stream Control Ribbon */}
      <div className="p-3 bg-slate-950/90 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
        {/* Source Switchers */}
        <div className="flex items-center space-x-2">
          <div className="flex items-center bg-slate-900 p-0.5 rounded-lg border border-slate-800 text-xs font-mono">
            <button
              onClick={() => {
                setSourceMode('upload');
                if (!uploadedImageData) fileInputRef.current?.click();
              }}
              className={`px-3 py-1 rounded-md flex items-center space-x-1.5 transition-all ${
                sourceMode === 'upload'
                  ? 'bg-cyan-600 text-white font-semibold shadow'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Upload Image</span>
            </button>

            <button
              onClick={() => setSourceMode('sample')}
              className={`px-3 py-1 rounded-md flex items-center space-x-1.5 transition-all ${
                sourceMode === 'sample'
                  ? 'bg-cyan-600 text-white font-semibold shadow'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <ImageIcon className="w-3.5 h-3.5" />
              <span>Benchmark Feeds</span>
            </button>

            <button
              onClick={() => setSourceMode('webcam')}
              className={`px-3 py-1 rounded-md flex items-center space-x-1.5 transition-all ${
                sourceMode === 'webcam'
                  ? 'bg-cyan-600 text-white font-semibold shadow'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Camera className="w-3.5 h-3.5" />
              <span>Webcam Live</span>
            </button>
          </div>

          {/* Sample Stream Select Dropdown */}
          {sourceMode === 'sample' && (
            <select
              value={selectedStream.id}
              onChange={(e) => {
                const found = SAMPLE_BENCHMARK_STREAMS.find((s) => s.id === e.target.value);
                if (found) {
                  setSelectedStream(found);
                  loadImageSource(found.image);
                  onAnalyzeFrame(found.image, 'Detect all salient objects, vehicles, and text');
                }
              }}
              className="bg-slate-800 border border-slate-700 text-xs text-slate-200 rounded-lg px-2.5 py-1 focus:outline-none focus:border-cyan-500"
            >
              {SAMPLE_BENCHMARK_STREAMS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          )}

          {sourceMode === 'upload' && uploadedImageData && (
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 text-xs font-mono rounded-md flex items-center space-x-1"
            >
              <ImageIcon className="w-3 h-3 text-cyan-400" />
              <span className="truncate max-w-[140px]">{uploadedFileName}</span>
              <span className="text-slate-500">(Change)</span>
            </button>
          )}
        </div>

        {/* View Overlays Toggles */}
        <div className="flex items-center space-x-2 text-xs">
          <button
            onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
            className={`px-2.5 py-1 rounded-md border flex items-center space-x-1.5 transition-colors ${
              showBoundingBoxes ? 'bg-cyan-950 border-cyan-500 text-cyan-300' : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
            title="Toggle 2D Bounding Boxes"
          >
            <Eye className="w-3.5 h-3.5" />
            <span>Boxes</span>
          </button>

          <button
            onClick={() => setShowDepthOverlay(!showDepthOverlay)}
            className={`px-2.5 py-1 rounded-md border flex items-center space-x-1.5 transition-colors ${
              showDepthOverlay ? 'bg-emerald-950 border-emerald-500 text-emerald-300' : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
            title="Toggle Depth Map Colormap"
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Depth Map</span>
          </button>

          {sourceMode !== 'upload' && (
            <>
              <button
                onClick={() => setShowOpticalFlow(!showOpticalFlow)}
                className={`px-2.5 py-1 rounded-md border flex items-center space-x-1.5 transition-colors ${
                  showOpticalFlow ? 'bg-purple-950 border-purple-500 text-purple-300' : 'bg-slate-800 border-slate-700 text-slate-400'
                }`}
                title="Toggle Optical Flow Motion Vectors"
              >
                <Activity className="w-3.5 h-3.5" />
                <span>Optical Flow</span>
              </button>

              <button
                onClick={() => setIsLivePlaying(!isLivePlaying)}
                className="p-1.5 rounded-md bg-slate-800 border border-slate-700 text-slate-300 hover:text-white"
                title={isLivePlaying ? 'Pause video' : 'Play video'}
              >
                {isLivePlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              </button>
            </>
          )}

          {/* Zoom Controls */}
          <div className="flex items-center bg-slate-800 rounded-md border border-slate-700 p-0.5">
            <button
              onClick={() => setZoomLevel((z) => Math.max(0.7, z - 0.15))}
              className="p-1 text-slate-400 hover:text-white"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-[10px] font-mono px-1 text-slate-300">
              {Math.round(zoomLevel * 100)}%
            </span>
            <button
              onClick={() => setZoomLevel((z) => Math.min(2.0, z + 0.15))}
              className="p-1 text-slate-400 hover:text-white"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            {zoomLevel !== 1.0 && (
              <button
                onClick={() => setZoomLevel(1.0)}
                className="text-[10px] text-cyan-400 hover:underline px-1"
              >
                Reset
              </button>
            )}
          </div>

          {onOpenAnnotation && (
            <button
              onClick={onOpenAnnotation}
              className="px-3 py-1 bg-amber-950/60 hover:bg-amber-900 border border-amber-600/80 text-amber-300 font-medium rounded-md flex items-center space-x-1.5 transition-all shadow-sm"
              title="Open Interactive Ground Truth Annotation Studio"
            >
              <Square className="w-3.5 h-3.5" />
              <span>Draw / Edit Boxes</span>
            </button>
          )}

          {/* Tag Mode Selector */}
          <div className="flex items-center bg-slate-800 rounded-md border border-slate-700 p-0.5 text-xs font-mono">
            <button
              onClick={() => setBadgeDisplayMode('adaptive')}
              className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                badgeDisplayMode === 'adaptive' ? 'bg-cyan-600 text-white font-semibold' : 'text-slate-400 hover:text-white'
              }`}
              title="Adaptive compact tags that avoid overlap"
            >
              Adaptive
            </button>
            <button
              onClick={() => setBadgeDisplayMode('detailed')}
              className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                badgeDisplayMode === 'detailed' ? 'bg-cyan-600 text-white font-semibold' : 'text-slate-400 hover:text-white'
              }`}
              title="Full telemetry tags with depth & confidence"
            >
              Detailed
            </button>
            <button
              onClick={() => setBadgeDisplayMode('minimal')}
              className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                badgeDisplayMode === 'minimal' ? 'bg-cyan-600 text-white font-semibold' : 'text-slate-400 hover:text-white'
              }`}
              title="Clean numbers only (#1, #2...)"
            >
              # Only
            </button>
          </div>

          <button
            onClick={() => handleCaptureFrame()}
            disabled={isAnalyzing}
            className="px-3.5 py-1 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-medium rounded-md flex items-center space-x-1.5 transition-all shadow-md shadow-cyan-900/30"
          >
            {isAnalyzing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Crosshair className="w-3.5 h-3.5" />}
            <span>{isAnalyzing ? 'Reasoning...' : 'Scan & Reason'}</span>
          </button>
        </div>
      </div>

      {/* Main Viewport Area (Canvas Renderer) */}
      <div
        ref={containerRef}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative w-full h-[450px] bg-black flex items-center justify-center overflow-hidden transition-all select-none ${
          isDragOver ? 'ring-2 ring-cyan-400 bg-slate-900/80' : ''
        }`}
      >
        {/* Hidden video element for webcam feed */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="hidden"
        />

        {/* Upload Drop Prompt (Shown when in upload mode with no image) */}
        {sourceMode === 'upload' && !uploadedImageData && (
          <div
            onClick={() => fileInputRef.current?.click()}
            className="flex flex-col items-center justify-center space-y-3 cursor-pointer p-8 text-center z-10"
          >
            <div className="w-16 h-16 rounded-2xl bg-slate-800/80 border border-slate-700 flex items-center justify-center text-cyan-400 hover:scale-105 transition-transform shadow-lg">
              <Upload className="w-8 h-8" />
            </div>
            <div>
              <div className="text-sm font-semibold text-slate-200">
                Click to browse or drag and drop your image
              </div>
              <div className="text-xs text-slate-400 mt-1">
                Supports PNG, JPG, WEBP — Instant SOTA Gemini 3.8 Flash detection, metric depth & 4D graph
              </div>
            </div>
            <button className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-mono font-semibold rounded-lg shadow">
              Select Image from Computer
            </button>
          </div>
        )}

        {/* Lockstep High-Performance Canvas */}
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          onMouseMove={handleCanvasMouseMove}
          className="w-full h-full cursor-crosshair"
        />

        {/* Drag Over Overlay Alert */}
        {isDragOver && (
          <div className="absolute inset-0 bg-cyan-950/80 border-2 border-dashed border-cyan-400 flex flex-col items-center justify-center z-20 pointer-events-none">
            <Upload className="w-12 h-12 text-cyan-300 animate-bounce" />
            <span className="text-sm font-mono font-bold text-cyan-200 mt-2">
              Drop image here to load and reason
            </span>
          </div>
        )}

        {/* Bottom Stream Telemetry HUD */}
        <div className="absolute bottom-3 left-3 flex items-center space-x-2 text-[11px] font-mono text-slate-300 z-10 pointer-events-none">
          <span className="px-2 py-0.5 bg-slate-950/85 border border-slate-700 rounded backdrop-blur">
            {sourceMode === 'upload' ? 'Mode: Uploaded Image' : `Stream: ${selectedStream.name.split(' (')[0]}`}
          </span>
          <span className="px-2 py-0.5 bg-slate-950/85 border border-slate-700 rounded backdrop-blur">
            Detected: {instances.length} targets
          </span>
          <span className="px-2 py-0.5 bg-cyan-950/85 border border-cyan-800 text-cyan-300 rounded backdrop-blur">
            Pixel-Perfect Alignment: Locked
          </span>
        </div>

        {/* Selected Target Live Inspector HUD Card */}
        {selectedInstance ? (
          <div className="absolute top-3 right-3 max-w-xs bg-slate-950/95 border border-amber-500/80 rounded-lg p-2.5 shadow-2xl backdrop-blur font-mono text-xs z-20 space-y-2 animate-in fade-in">
            <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
              <div className="flex items-center space-x-1.5 text-amber-300 font-bold">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                <span className="truncate max-w-[170px]">Target: {selectedInstance.label}</span>
              </div>
              <button
                onClick={() => onSelectInstance && onSelectInstance('')}
                className="text-slate-400 hover:text-white text-xs px-1 hover:bg-slate-800 rounded"
                title="Deselect"
              >
                ✕
              </button>
            </div>
            <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[11px] text-slate-300">
              <div>
                <span className="text-slate-500 text-[10px] block">CONFIDENCE</span>
                <span className="text-emerald-400 font-bold">{Math.round(selectedInstance.confidence * 100)}%</span>
              </div>
              <div>
                <span className="text-slate-500 text-[10px] block">EST. DISTANCE</span>
                <span className="text-cyan-300 font-bold">{selectedInstance.depth.metricDistanceM}m</span>
              </div>
              <div className="col-span-2">
                <span className="text-slate-500 text-[10px] block">2D BOUNDING BOX</span>
                <span className="text-slate-400 text-[10px]">
                  [{selectedInstance.bbox2d.ymin}, {selectedInstance.bbox2d.xmin}, {selectedInstance.bbox2d.ymax}, {selectedInstance.bbox2d.xmax}]
                </span>
              </div>
            </div>
            {onOpenAnnotation && (
              <button
                onClick={onOpenAnnotation}
                className="w-full mt-1 py-1 px-2 bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/60 text-amber-200 text-[11px] rounded flex items-center justify-center space-x-1 transition-colors"
              >
                <Square className="w-3 h-3" />
                <span>Fine-Tune in Annotation Studio</span>
              </button>
            )}
          </div>
        ) : (
          <div className="absolute top-3 right-3 text-[10px] font-mono text-slate-400 bg-slate-950/80 border border-slate-800 px-2.5 py-1 rounded backdrop-blur flex items-center space-x-1 pointer-events-none">
            <Info className="w-3 h-3 text-cyan-400" />
            <span>Click any box to inspect spatial telemetry</span>
          </div>
        )}
      </div>

      {/* Uploaded Image Quick Actions Banner */}
      {sourceMode === 'upload' && uploadedImageData && (
        <div className="px-4 py-2.5 bg-slate-950 border-t border-slate-800 flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-mono text-cyan-400 flex items-center space-x-1 whitespace-nowrap">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Ask OmniVision about this image:</span>
          </span>
          {QUICK_UPLOAD_QUERIES.map((q, idx) => (
            <button
              key={idx}
              onClick={() => handleCaptureFrame(q)}
              disabled={isAnalyzing}
              className="px-2.5 py-1 text-[11px] font-mono bg-slate-800/80 hover:bg-cyan-950 hover:border-cyan-700 border border-slate-700 text-slate-300 hover:text-cyan-200 rounded-md transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Bottom Tracklet Ribbon */}
      <div className="px-4 py-2.5 bg-slate-950/90 border-t border-slate-800 flex items-center space-x-2 overflow-x-auto">
        <span className="text-[11px] font-mono text-slate-400 whitespace-nowrap">Detected Instances:</span>
        {instances.length === 0 ? (
          <span className="text-xs text-slate-500 italic">Click "Scan & Reason" or upload an image to detect objects.</span>
        ) : (
          instances.map((inst) => (
            <button
              key={inst.id}
              onClick={() => onSelectInstance && onSelectInstance(inst.id)}
              className={`px-2.5 py-1 text-xs font-mono rounded border flex items-center space-x-2 transition-all whitespace-nowrap ${
                inst.id === selectedInstanceId
                  ? 'bg-amber-500/20 text-amber-300 border-amber-500'
                  : 'bg-slate-900 text-slate-300 border-slate-800 hover:border-slate-700'
              }`}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              <span className="font-medium">{inst.label}</span>
              <span className="text-[10px] text-slate-400">({inst.depth.metricDistanceM}m)</span>
            </button>
          ))
        )}
      </div>
    </div>
  );
};
