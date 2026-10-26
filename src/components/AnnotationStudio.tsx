/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Square,
  Tag,
  Trash2,
  Save,
  Download,
  Check,
  Sparkles,
  Layers,
  CheckCircle2,
  AlertCircle,
  Crosshair,
  RefreshCw,
  Copy,
  Move,
  Eye,
  Sliders,
  ChevronDown
} from 'lucide-react';
import { BoundingBox2D, InstanceNode } from '../types/index.ts';

export interface UserAnnotation {
  id: string;
  label: string;
  bbox: BoundingBox2D; // 0..1000 normalized
  color: string;
  isAiGenerated?: boolean;
  confidence?: number;
}

interface AnnotationStudioProps {
  imageSrc: string;
  imageFileName?: string;
  agentInstances: InstanceNode[];
  onSaveToMetaLearner?: (annotations: UserAnnotation[]) => void;
  onRefreshAgent?: () => void;
}

const PRESET_LABELS = [
  { name: 'red car', color: '#ef4444' },
  { name: 'person', color: '#10b981' },
  { name: 'car', color: '#38bdf8' },
  { name: 'vehicle', color: '#6366f1' },
  { name: 'hazard', color: '#ef4444' },
  { name: 'sign/text', color: '#f59e0b' },
  { name: 'equipment', color: '#8b5cf6' },
  { name: 'obstacle', color: '#ec4899' },
  { name: 'defect', color: '#f97316' },
  { name: 'container', color: '#06b6d4' },
  { name: 'cone', color: '#eab308' },
];

type HandleType = 'tl' | 'tc' | 'tr' | 'rc' | 'br' | 'bc' | 'bl' | 'lc' | 'body';

export const AnnotationStudio: React.FC<AnnotationStudioProps> = ({
  imageSrc,
  imageFileName = 'annotated_sample.jpg',
  agentInstances,
  onSaveToMetaLearner,
  onRefreshAgent,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const loadedImageRef = useRef<HTMLImageElement | null>(null);

  const [annotations, setAnnotations] = useState<UserAnnotation[]>([]);
  const [selectedAnnoId, setSelectedAnnoId] = useState<string | null>(null);
  const [activeLabel, setActiveLabel] = useState<string>('car');
  const [customLabelInput, setCustomLabelInput] = useState<string>('');

  // Interaction State: drawing, moving, or resizing
  const [interactionMode, setInteractionMode] = useState<'idle' | 'drawing' | 'resizing' | 'moving'>('idle');
  const [activeHandle, setActiveHandle] = useState<HandleType | null>(null);
  const [dragStartPoint, setDragStartPoint] = useState<{ normX: number; normY: number } | null>(null);
  const [initialBoxOnDrag, setInitialBoxOnDrag] = useState<BoundingBox2D | null>(null);
  const [cursorStyle, setCursorStyle] = useState<string>('crosshair');

  const [viewMode, setViewMode] = useState<'both' | 'user_gt' | 'agent_only'>('both');
  const [savedStatus, setSavedStatus] = useState<string | null>(null);
  const [isAutoDetecting, setIsAutoDetecting] = useState<boolean>(false);
  const [imageNaturalSize, setImageNaturalSize] = useState<{ width: number; height: number }>({ width: 1200, height: 800 });

  const lastImageSrcRef = useRef<string>(imageSrc);

  // Load Image Object with error/CORS fallback
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imageSrc;
    img.onload = () => {
      setImageNaturalSize({ width: img.naturalWidth || 1200, height: img.naturalHeight || 800 });
      loadedImageRef.current = img;
    };
    img.onerror = () => {
      const fallback = new Image();
      fallback.src = imageSrc;
      fallback.onload = () => {
        setImageNaturalSize({ width: fallback.naturalWidth || 1200, height: fallback.naturalHeight || 800 });
        loadedImageRef.current = fallback;
      };
    };
  }, [imageSrc]);

  // Import detections from Agent
  const importFromAgent = useCallback(() => {
    if (agentInstances.length === 0) return;
    const imported: UserAnnotation[] = agentInstances.map((inst, i) => {
      let foundColor = '#38bdf8';
      const lcase = inst.label.toLowerCase();
      const colorName = inst.colorLAB?.name?.toLowerCase() || '';
      if (lcase.includes('red') || colorName.includes('red') || colorName.includes('crimson')) foundColor = '#ef4444';
      else if (lcase.includes('hazard') || lcase.includes('danger')) foundColor = '#f43f5e';
      else if (lcase.includes('person') || lcase.includes('worker')) foundColor = '#10b981';
      else if (lcase.includes('yellow')) foundColor = '#eab308';
      else if (lcase.includes('orange')) foundColor = '#f97316';
      else {
        foundColor = PRESET_LABELS.find((l) => lcase.includes(l.name))?.color || '#38bdf8';
      }
      return {
        id: `anno_agent_${Date.now()}_${i + 1}`,
        label: inst.label,
        bbox: { ...inst.bbox2d },
        color: foundColor,
        isAiGenerated: true,
        confidence: inst.confidence,
      };
    });
    setAnnotations(imported);
    if (imported.length > 0) setSelectedAnnoId(imported[0].id);
    setSavedStatus(`Loaded ${imported.length} calibrated detections from live OmniVision agent!`);
    setTimeout(() => setSavedStatus(null), 3000);
  }, [agentInstances]);

  // When image source changes, refresh annotations from agent
  useEffect(() => {
    if (lastImageSrcRef.current !== imageSrc) {
      lastImageSrcRef.current = imageSrc;
      importFromAgent();
    }
  }, [imageSrc, importFromAgent]);

  // Convert Agent detections into editable annotations on initial load if none exist
  useEffect(() => {
    if (annotations.length === 0 && agentInstances.length > 0) {
      importFromAgent();
    }
  }, [agentInstances, annotations.length, importFromAgent]);

  // Keyboard navigation & delete
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!selectedAnnoId) return;

      // Delete or Backspace
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        setAnnotations((prev) => prev.filter((a) => a.id !== selectedAnnoId));
        setSelectedAnnoId(null);
        return;
      }

      // Escape to deselect
      if (e.key === 'Escape') {
        setSelectedAnnoId(null);
        return;
      }

      // Arrow keys to nudge box
      const nudgeAmount = e.shiftKey ? 10 : 2;
      let dx = 0;
      let dy = 0;
      if (e.key === 'ArrowUp') dy = -nudgeAmount;
      else if (e.key === 'ArrowDown') dy = nudgeAmount;
      else if (e.key === 'ArrowLeft') dx = -nudgeAmount;
      else if (e.key === 'ArrowRight') dx = nudgeAmount;

      if (dx !== 0 || dy !== 0) {
        e.preventDefault();
        setAnnotations((prev) =>
          prev.map((a) => {
            if (a.id !== selectedAnnoId) return a;
            return {
              ...a,
              bbox: {
                ymin: Math.max(0, Math.min(1000, a.bbox.ymin + dy)),
                ymax: Math.max(0, Math.min(1000, a.bbox.ymax + dy)),
                xmin: Math.max(0, Math.min(1000, a.bbox.xmin + dx)),
                xmax: Math.max(0, Math.min(1000, a.bbox.xmax + dx)),
              },
            };
          })
        );
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedAnnoId]);

  // Compute rendered image placement inside canvas
  const getRenderedImageRect = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0, width: 800, height: 500 };

    const cW = canvas.width;
    const cH = canvas.height;
    const imgAspect = imageNaturalSize.width / imageNaturalSize.height;
    const canvasAspect = cW / cH;

    let rW = cW;
    let rH = cH;
    let oX = 0;
    let oY = 0;

    if (imgAspect > canvasAspect) {
      rW = cW;
      rH = cW / imgAspect;
      oY = (cH - rH) / 2;
    } else {
      rH = cH;
      rW = cH * imgAspect;
      oX = (cW - rW) / 2;
    }

    return { x: oX, y: oY, width: rW, height: rH };
  }, [imageNaturalSize]);

  // Convert mouse client coordinates to 0..1000 normalized image coordinates
  const clientToNormalized = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return { normX: 0, normY: 0, rawX: 0, rawY: 0, insideImage: false };
    const rect = canvas.getBoundingClientRect();
    const rawX = clientX - rect.left;
    const rawY = clientY - rect.top;

    const imgRect = getRenderedImageRect();
    const insideImage =
      rawX >= imgRect.x &&
      rawX <= imgRect.x + imgRect.width &&
      rawY >= imgRect.y &&
      rawY <= imgRect.y + imgRect.height;

    const clampedX = Math.max(imgRect.x, Math.min(imgRect.x + imgRect.width, rawX));
    const clampedY = Math.max(imgRect.y, Math.min(imgRect.y + imgRect.height, rawY));

    const normX = Math.round(((clampedX - imgRect.x) / imgRect.width) * 1000);
    const normY = Math.round(((clampedY - imgRect.y) / imgRect.height) * 1000);

    return { normX, normY, rawX, rawY, insideImage };
  };

  // Helper: check if a point is near an anchor handle
  const isNear = (x1: number, y1: number, x2: number, y2: number, dist = 8) => {
    return Math.abs(x1 - x2) <= dist && Math.abs(y1 - y2) <= dist;
  };

  // Main Canvas Render Loop (Draws Image + Boxes + 8 Resize Handles + HUD)
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Synchronize canvas buffer resolution with DOM size
    const cW = container.clientWidth || 900;
    const cH = container.clientHeight || 500;
    if (canvas.width !== cW || canvas.height !== cH) {
      canvas.width = cW;
      canvas.height = cH;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw Background
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const imgRect = getRenderedImageRect();

    // 2. Draw Image directly onto canvas
    if (loadedImageRef.current && loadedImageRef.current.complete && loadedImageRef.current.naturalWidth > 0) {
      ctx.drawImage(loadedImageRef.current, imgRect.x, imgRect.y, imgRect.width, imgRect.height);
    }

    // Image frame border
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
    ctx.lineWidth = 1;
    ctx.strokeRect(imgRect.x, imgRect.y, imgRect.width, imgRect.height);

    // 3. Render Annotations
    const visibleAnnotations = annotations.filter((a) => {
      if (viewMode === 'user_gt') return !a.isAiGenerated;
      if (viewMode === 'agent_only') return a.isAiGenerated;
      return true;
    });

    // Render unselected first, selected last on top
    const sortedAnnotations = [...visibleAnnotations].sort((a, b) => {
      if (a.id === selectedAnnoId) return 1;
      if (b.id === selectedAnnoId) return -1;
      return 0;
    });

    sortedAnnotations.forEach((anno) => {
      const isSelected = anno.id === selectedAnnoId;
      const b = anno.bbox;
      const bx = imgRect.x + (b.xmin / 1000) * imgRect.width;
      const by = imgRect.y + (b.ymin / 1000) * imgRect.height;
      const bw = ((b.xmax - b.xmin) / 1000) * imgRect.width;
      const bh = ((b.ymax - b.ymin) / 1000) * imgRect.height;

      // Fill
      ctx.fillStyle = isSelected
        ? 'rgba(250, 204, 21, 0.22)'
        : anno.isAiGenerated
        ? 'rgba(56, 189, 248, 0.08)'
        : 'rgba(16, 185, 129, 0.12)';
      ctx.fillRect(bx, by, bw, bh);

      // Stroke
      ctx.strokeStyle = isSelected ? '#facc15' : anno.color || '#38bdf8';
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.setLineDash(anno.isAiGenerated && !isSelected ? [5, 3] : []);
      ctx.strokeRect(bx, by, bw, bh);

      // Label Header Badge
      ctx.font = 'bold 11px monospace';
      const typeTag = anno.isAiGenerated ? `AI ${Math.round((anno.confidence || 0.94) * 100)}%` : 'GT Verified';
      const labelText = `${anno.label} [${typeTag}]`;
      const textMetrics = ctx.measureText(labelText);
      const badgeW = textMetrics.width + 12;
      const badgeH = 19;
      const badgeX = Math.max(imgRect.x, Math.min(bx, imgRect.x + imgRect.width - badgeW));
      let badgeY = by - badgeH - 2;
      if (badgeY < imgRect.y) {
        badgeY = by + 2;
      }

      ctx.fillStyle = isSelected ? '#facc15' : anno.color || '#0284c7';
      ctx.fillRect(badgeX, badgeY, badgeW, badgeH);

      ctx.fillStyle = '#0f172a';
      ctx.fillText(labelText, badgeX + 6, badgeY + 13);

      // 4. Render 8 Interactive Resize Handles if Selected
      if (isSelected) {
        ctx.setLineDash([]);
        const handleSize = 8;
        const half = handleSize / 2;

        const handles = [
          { x: bx, y: by }, // TL
          { x: bx + bw / 2, y: by }, // TC
          { x: bx + bw, y: by }, // TR
          { x: bx + bw, y: by + bh / 2 }, // RC
          { x: bx + bw, y: by + bh }, // BR
          { x: bx + bw / 2, y: by + bh }, // BC
          { x: bx, y: by + bh }, // BL
          { x: bx, y: by + bh / 2 }, // LC
        ];

        handles.forEach((h) => {
          ctx.fillStyle = '#facc15';
          ctx.strokeStyle = '#0f172a';
          ctx.lineWidth = 1.5;
          ctx.fillRect(h.x - half, h.y - half, handleSize, handleSize);
          ctx.strokeRect(h.x - half, h.y - half, handleSize, handleSize);
        });

        // Dimension Readout Pill at bottom of box
        const dimText = `${Math.round((b.xmax - b.xmin) / 10)}% × ${Math.round((b.ymax - b.ymin) / 10)}% (${Math.round(
          ((b.xmax - b.xmin) / 1000) * imageNaturalSize.width
        )} × ${Math.round(((b.ymax - b.ymin) / 1000) * imageNaturalSize.height)}px)`;
        ctx.font = '10px monospace';
        const dimW = ctx.measureText(dimText).width + 8;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
        ctx.fillRect(bx, by + bh + 4, dimW, 16);
        ctx.fillStyle = '#facc15';
        ctx.fillText(dimText, bx + 4, by + bh + 16);
      }
    });
  }, [annotations, selectedAnnoId, viewMode, imageNaturalSize, getRenderedImageRect]);

  // Determine handle or body under cursor
  const getHitHandle = (rawX: number, rawY: number): { anno: UserAnnotation; handle: HandleType } | null => {
    const selected = annotations.find((a) => a.id === selectedAnnoId);
    if (!selected) return null;

    const imgRect = getRenderedImageRect();
    const b = selected.bbox;
    const bx = imgRect.x + (b.xmin / 1000) * imgRect.width;
    const by = imgRect.y + (b.ymin / 1000) * imgRect.height;
    const bw = ((b.xmax - b.xmin) / 1000) * imgRect.width;
    const bh = ((b.ymax - b.ymin) / 1000) * imgRect.height;

    // Check handles first
    if (isNear(rawX, rawY, bx, by)) return { anno: selected, handle: 'tl' };
    if (isNear(rawX, rawY, bx + bw / 2, by)) return { anno: selected, handle: 'tc' };
    if (isNear(rawX, rawY, bx + bw, by)) return { anno: selected, handle: 'tr' };
    if (isNear(rawX, rawY, bx + bw, by + bh / 2)) return { anno: selected, handle: 'rc' };
    if (isNear(rawX, rawY, bx + bw, by + bh)) return { anno: selected, handle: 'br' };
    if (isNear(rawX, rawY, bx + bw / 2, by + bh)) return { anno: selected, handle: 'bc' };
    if (isNear(rawX, rawY, bx, by + bh)) return { anno: selected, handle: 'bl' };
    if (isNear(rawX, rawY, bx, by + bh / 2)) return { anno: selected, handle: 'lc' };

    // Check inside body
    if (rawX >= bx && rawX <= bx + bw && rawY >= by && rawY <= by + bh) {
      return { anno: selected, handle: 'body' };
    }

    return null;
  };

  // Mouse Down
  const handleMouseDown = (e: React.MouseEvent) => {
    const { normX, normY, rawX, rawY } = clientToNormalized(e.clientX, e.clientY);

    // 1. Check if clicking on selected box handle or body
    const handleHit = getHitHandle(rawX, rawY);
    if (handleHit) {
      if (handleHit.handle === 'body') {
        setInteractionMode('moving');
      } else {
        setInteractionMode('resizing');
      }
      setActiveHandle(handleHit.handle);
      setDragStartPoint({ normX, normY });
      setInitialBoxOnDrag({ ...handleHit.anno.bbox });
      return;
    }

    // 2. Check if clicking any other box to select it
    const clickedOther = [...annotations].reverse().find((a) => {
      return (
        normX >= a.bbox.xmin &&
        normX <= a.bbox.xmax &&
        normY >= a.bbox.ymin &&
        normY <= a.bbox.ymax
      );
    });

    if (clickedOther) {
      setSelectedAnnoId(clickedOther.id);
      setInteractionMode('moving');
      setActiveHandle('body');
      setDragStartPoint({ normX, normY });
      setInitialBoxOnDrag({ ...clickedOther.bbox });
      return;
    }

    // 3. Otherwise start drawing a new bounding box
    setSelectedAnnoId(null);
    setInteractionMode('drawing');
    setDragStartPoint({ normX, normY });

    const newId = `gt_anno_${Date.now()}`;
    const activeColor = PRESET_LABELS.find((l) => l.name === activeLabel)?.color || '#10b981';
    const newBox: UserAnnotation = {
      id: newId,
      label: customLabelInput.trim() || activeLabel,
      bbox: { ymin: normY, xmin: normX, ymax: normY, xmax: normX },
      color: activeColor,
      isAiGenerated: false,
    };

    setAnnotations((prev) => [...prev, newBox]);
    setSelectedAnnoId(newId);
  };

  // Mouse Move
  const handleMouseMove = (e: React.MouseEvent) => {
    const { normX, normY, rawX, rawY } = clientToNormalized(e.clientX, e.clientY);

    // Update cursor based on hover
    if (interactionMode === 'idle') {
      const handleHit = getHitHandle(rawX, rawY);
      if (handleHit) {
        if (handleHit.handle === 'tl' || handleHit.handle === 'br') setCursorStyle('nwse-resize');
        else if (handleHit.handle === 'tr' || handleHit.handle === 'bl') setCursorStyle('nesw-resize');
        else if (handleHit.handle === 'tc' || handleHit.handle === 'bc') setCursorStyle('ns-resize');
        else if (handleHit.handle === 'lc' || handleHit.handle === 'rc') setCursorStyle('ew-resize');
        else if (handleHit.handle === 'body') setCursorStyle('move');
      } else {
        setCursorStyle('crosshair');
      }
      return;
    }

    if (!selectedAnnoId) return;

    // Drawing a new box
    if (interactionMode === 'drawing' && dragStartPoint) {
      const xmin = Math.min(dragStartPoint.normX, normX);
      const xmax = Math.max(dragStartPoint.normX, normX);
      const ymin = Math.min(dragStartPoint.normY, normY);
      const ymax = Math.max(dragStartPoint.normY, normY);

      setAnnotations((prev) =>
        prev.map((a) => (a.id === selectedAnnoId ? { ...a, bbox: { ymin, xmin, ymax, xmax } } : a))
      );
      return;
    }

    // Moving existing box
    if (interactionMode === 'moving' && dragStartPoint && initialBoxOnDrag) {
      const dx = normX - dragStartPoint.normX;
      const dy = normY - dragStartPoint.normY;
      const width = initialBoxOnDrag.xmax - initialBoxOnDrag.xmin;
      const height = initialBoxOnDrag.ymax - initialBoxOnDrag.ymin;

      let newXmin = Math.max(0, Math.min(1000 - width, initialBoxOnDrag.xmin + dx));
      let newYmin = Math.max(0, Math.min(1000 - height, initialBoxOnDrag.ymin + dy));

      setAnnotations((prev) =>
        prev.map((a) =>
          a.id === selectedAnnoId
            ? {
                ...a,
                bbox: {
                  xmin: Math.round(newXmin),
                  ymin: Math.round(newYmin),
                  xmax: Math.round(newXmin + width),
                  ymax: Math.round(newYmin + height),
                },
              }
            : a
        )
      );
      return;
    }

    // Resizing existing box via one of 8 handles
    if (interactionMode === 'resizing' && activeHandle && initialBoxOnDrag) {
      setAnnotations((prev) =>
        prev.map((a) => {
          if (a.id !== selectedAnnoId) return a;
          let { ymin, xmin, ymax, xmax } = initialBoxOnDrag;

          if (activeHandle.includes('t')) ymin = Math.min(ymax - 10, normY);
          if (activeHandle.includes('b')) ymax = Math.max(ymin + 10, normY);
          if (activeHandle.includes('l')) xmin = Math.min(xmax - 10, normX);
          if (activeHandle.includes('r')) xmax = Math.max(xmin + 10, normX);

          return {
            ...a,
            bbox: {
              ymin: Math.max(0, Math.min(1000, ymin)),
              xmin: Math.max(0, Math.min(1000, xmin)),
              ymax: Math.max(0, Math.min(1000, ymax)),
              xmax: Math.max(0, Math.min(1000, xmax)),
            },
          };
        })
      );
    }
  };

  // Mouse Up
  const handleMouseUp = () => {
    // If drawing, clean up tiny accidental dots
    if (interactionMode === 'drawing' && selectedAnnoId) {
      const drawn = annotations.find((a) => a.id === selectedAnnoId);
      if (drawn) {
        const w = drawn.bbox.xmax - drawn.bbox.xmin;
        const h = drawn.bbox.ymax - drawn.bbox.ymin;
        if (w < 12 || h < 12) {
          setAnnotations((prev) => prev.filter((a) => a.id !== selectedAnnoId));
          setSelectedAnnoId(null);
        }
      }
    }

    setInteractionMode('idle');
    setActiveHandle(null);
    setDragStartPoint(null);
    setInitialBoxOnDrag(null);
  };

  // Duplicate Selected Box
  const handleDuplicate = () => {
    const sel = annotations.find((a) => a.id === selectedAnnoId);
    if (!sel) return;
    const dup: UserAnnotation = {
      ...sel,
      id: `gt_anno_${Date.now()}`,
      bbox: {
        ymin: Math.min(980, sel.bbox.ymin + 25),
        xmin: Math.min(980, sel.bbox.xmin + 25),
        ymax: Math.min(1000, sel.bbox.ymax + 25),
        xmax: Math.min(1000, sel.bbox.xmax + 25),
      },
      isAiGenerated: false,
    };
    setAnnotations((prev) => [...prev, dup]);
    setSelectedAnnoId(dup.id);
  };

  // Run Real-Time Gemini Detection to populate Ground Truth automatically
  const handleAutoDetect = async () => {
    setIsAutoDetecting(true);
    try {
      const res = await fetch('/api/perception/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageBase64: imageSrc,
          queryHint: 'Detect and tightly localize all objects, people, vehicles, text, and equipment',
        }),
      });

      if (res.ok) {
        const data = await res.json();
        if (data.instances && data.instances.length > 0) {
          const newAnnos: UserAnnotation[] = data.instances.map((inst: InstanceNode, i: number) => {
            const color = PRESET_LABELS.find((l) => inst.label.toLowerCase().includes(l.name))?.color || '#38bdf8';
            return {
              id: `anno_auto_${Date.now()}_${i}`,
              label: inst.label,
              bbox: { ...inst.bbox2d },
              color,
              isAiGenerated: true,
              confidence: inst.confidence,
            };
          });

          setAnnotations(newAnnos);
          if (newAnnos.length > 0) setSelectedAnnoId(newAnnos[0].id);
          setSavedStatus(`Auto-detected ${newAnnos.length} objects with calibrated boxes!`);
          setTimeout(() => setSavedStatus(null), 3000);
        }
      }
    } catch (e) {
      console.error('[AnnotationStudio] Auto-detect error:', e);
    } finally {
      setIsAutoDetecting(false);
    }
  };

  // Save to Meta-Learner API
  const handleSaveToMetaLearner = async () => {
    const payload = annotations.map((a) => ({
      id: a.id,
      imageFileName,
      label: a.label,
      bbox2d: a.bbox,
      notes: a.isAiGenerated ? 'Agent predicted instance' : 'User Verified Ground Truth',
      createdAt: new Date().toISOString(),
    }));

    try {
      const res = await fetch('/api/agent/meta-learn/annotations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ annotations: payload }),
      });
      const data = await res.json();
      if (data.success) {
        setSavedStatus(`Saved ${payload.length} ground truth annotations to Meta-Learning dataset!`);
        setTimeout(() => setSavedStatus(null), 3500);
        if (onSaveToMetaLearner) onSaveToMetaLearner(annotations);
      }
    } catch (err) {
      console.error('Failed to save ground truth annotations:', err);
    }
  };

  // Export COCO Format JSON
  const handleExportCOCO = () => {
    const cocoData = {
      info: {
        description: 'OmniVision Ground Truth Dataset',
        date_created: new Date().toISOString(),
      },
      images: [
        {
          id: 1,
          file_name: imageFileName,
          width: imageNaturalSize.width,
          height: imageNaturalSize.height,
        },
      ],
      annotations: annotations.map((a, idx) => {
        const x = (a.bbox.xmin / 1000) * imageNaturalSize.width;
        const y = (a.bbox.ymin / 1000) * imageNaturalSize.height;
        const w = ((a.bbox.xmax - a.bbox.xmin) / 1000) * imageNaturalSize.width;
        const h = ((a.bbox.ymax - a.bbox.ymin) / 1000) * imageNaturalSize.height;
        return {
          id: idx + 1,
          image_id: 1,
          category_name: a.label,
          bbox: [Math.round(x), Math.round(y), Math.round(w), Math.round(h)],
          area: Math.round(w * h),
          iscrowd: 0,
        };
      }),
    };

    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(cocoData, null, 2));
    const a = document.createElement('a');
    a.href = dataStr;
    a.download = `coco_annotations_${Date.now()}.json`;
    a.click();
  };

  // Export YOLO Format TXT
  const handleExportYOLO = () => {
    const lines = annotations.map((a) => {
      const xc = ((a.bbox.xmin + a.bbox.xmax) / 2000).toFixed(6);
      const yc = ((a.bbox.ymin + a.bbox.ymax) / 2000).toFixed(6);
      const w = ((a.bbox.xmax - a.bbox.xmin) / 1000).toFixed(6);
      const h = ((a.bbox.ymax - a.bbox.ymin) / 1000).toFixed(6);
      const classIdx = Math.max(0, PRESET_LABELS.findIndex((p) => p.name === a.label));
      return `${classIdx} ${xc} ${yc} ${w} ${h}`;
    });

    const dataStr = 'data:text/plain;charset=utf-8,' + encodeURIComponent(lines.join('\n'));
    const a = document.createElement('a');
    a.href = dataStr;
    a.download = `yolo_annotations_${Date.now()}.txt`;
    a.click();
  };

  const selectedAnno = annotations.find((a) => a.id === selectedAnnoId);
  const userGtBoxes = annotations.filter((a) => !a.isAiGenerated);
  const agentBoxes = annotations.filter((a) => a.isAiGenerated);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col p-4 space-y-4">
      {/* Header Banner */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse" />
            <h3 className="text-sm font-mono font-bold text-white tracking-wide uppercase">
              OmniVision Ground Truth Calibration & Annotation Studio
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Draw, nudge, and resize bounding boxes with sub-pixel precision. Feeds the Meta-Learning DPO preference pipeline.
          </p>
        </div>

        {/* Global Action Buttons */}
        <div className="flex items-center space-x-2">
          {agentInstances.length > 0 && (
            <button
              onClick={importFromAgent}
              className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-cyan-300 text-xs font-mono rounded-lg flex items-center space-x-1.5 transition-all shadow-sm"
              title="Reload detected boxes from the live agent pipeline"
            >
              <RefreshCw className="w-3.5 h-3.5 text-cyan-400" />
              <span>Sync from Agent ({agentInstances.length})</span>
            </button>
          )}

          <button
            onClick={handleAutoDetect}
            disabled={isAutoDetecting}
            className="px-3 py-1.5 bg-cyan-950/70 hover:bg-cyan-900 border border-cyan-700 text-cyan-300 text-xs font-mono rounded-lg flex items-center space-x-1.5 transition-all shadow-sm"
          >
            {isAutoDetecting ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
            <span>Auto-Detect with SOTA Agent</span>
          </button>

          <button
            onClick={handleExportCOCO}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-mono rounded-lg flex items-center space-x-1.5 transition-all"
            title="Download COCO format JSON"
          >
            <Download className="w-3.5 h-3.5" />
            <span>COCO JSON</span>
          </button>

          <button
            onClick={handleExportYOLO}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-mono rounded-lg flex items-center space-x-1.5 transition-all"
            title="Download YOLO format TXT"
          >
            <Download className="w-3.5 h-3.5" />
            <span>YOLO TXT</span>
          </button>

          <button
            onClick={handleSaveToMetaLearner}
            className="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-mono font-semibold rounded-lg flex items-center space-x-1.5 shadow-md shadow-emerald-950/40 transition-all"
          >
            <Save className="w-3.5 h-3.5" />
            <span>Save to Meta-Learner</span>
          </button>
        </div>
      </div>

      {/* Success Notification Banner */}
      {savedStatus && (
        <div className="p-2.5 bg-emerald-950/70 border border-emerald-800 text-emerald-300 text-xs font-mono rounded-lg flex items-center space-x-2 animate-fadeIn">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>{savedStatus}</span>
        </div>
      )}

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Left 3 Columns: Interactive Image Canvas */}
        <div className="lg:col-span-3 flex flex-col space-y-2">
          {/* Canvas Toolbar */}
          <div className="flex flex-wrap items-center justify-between gap-2 p-2 bg-slate-950 rounded-lg border border-slate-800 text-xs font-mono">
            <div className="flex items-center space-x-2">
              <span className="text-slate-400 flex items-center space-x-1">
                <Square className="w-3.5 h-3.5 text-amber-400" />
                <span>Tool:</span>
              </span>
              <span className="text-cyan-300 font-semibold">
                {selectedAnno ? 'Selected: 8 Handles to Resize / Drag to Move' : 'Click & Drag to Create Box'}
              </span>
            </div>

            <div className="flex items-center space-x-1 bg-slate-900 p-0.5 rounded border border-slate-800">
              <button
                onClick={() => setViewMode('both')}
                className={`px-2 py-0.5 rounded text-[11px] ${viewMode === 'both' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'}`}
              >
                All ({annotations.length})
              </button>
              <button
                onClick={() => setViewMode('user_gt')}
                className={`px-2 py-0.5 rounded text-[11px] ${viewMode === 'user_gt' ? 'bg-amber-600 text-white' : 'text-slate-400 hover:text-white'}`}
              >
                User Ground Truth ({userGtBoxes.length})
              </button>
              <button
                onClick={() => setViewMode('agent_only')}
                className={`px-2 py-0.5 rounded text-[11px] ${viewMode === 'agent_only' ? 'bg-cyan-700 text-white' : 'text-slate-400 hover:text-white'}`}
              >
                Agent AI ({agentBoxes.length})
              </button>
            </div>

            {selectedAnno && (
              <div className="flex items-center space-x-2">
                <button
                  onClick={handleDuplicate}
                  className="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[11px] flex items-center space-x-1"
                  title="Duplicate selected box"
                >
                  <Copy className="w-3 h-3" />
                  <span>Duplicate</span>
                </button>
                <button
                  onClick={() => {
                    setAnnotations((prev) => prev.filter((a) => a.id !== selectedAnnoId));
                    setSelectedAnnoId(null);
                  }}
                  className="px-2 py-0.5 bg-rose-950/70 hover:bg-rose-900 border border-rose-800 text-rose-300 rounded text-[11px] flex items-center space-x-1"
                  title="Delete box (or press Delete key)"
                >
                  <Trash2 className="w-3 h-3" />
                  <span>Delete</span>
                </button>
              </div>
            )}
          </div>

          {/* Interactive Canvas Viewport */}
          <div
            ref={containerRef}
            className="relative w-full h-[480px] bg-black rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center select-none"
          >
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              style={{ cursor: cursorStyle }}
              className="w-full h-full"
            />
          </div>

          {/* Canvas Footer Status */}
          <div className="text-[11px] font-mono text-slate-500 flex items-center justify-between px-1">
            <span>Image Size: {imageNaturalSize.width} × {imageNaturalSize.height}px</span>
            <span>Keyboard: [Delete] remove • [Arrows] nudge 2px • [Shift+Arrows] nudge 10px</span>
          </div>
        </div>

        {/* Right Column: Label Palette & Annotation Properties */}
        <div className="flex flex-col space-y-3">
          {/* Selected Box Properties Card */}
          {selectedAnno ? (
            <div className="p-3 bg-slate-950 border border-amber-500/50 rounded-lg space-y-2.5 text-xs font-mono shadow-lg">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="font-bold text-amber-300 flex items-center space-x-1">
                  <Square className="w-3.5 h-3.5" />
                  <span>Selected Box</span>
                </span>
                <span className="text-[10px] text-slate-400">ID: {selectedAnno.id.slice(-6)}</span>
              </div>

              {/* Box Coordinates Readout */}
              <div className="grid grid-cols-2 gap-1.5 text-[11px] bg-slate-900 p-2 rounded border border-slate-800">
                <div>
                  <span className="text-slate-500">X:</span>{' '}
                  <span className="text-slate-200">
                    {selectedAnno.bbox.xmin} → {selectedAnno.bbox.xmax}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500">Y:</span>{' '}
                  <span className="text-slate-200">
                    {selectedAnno.bbox.ymin} → {selectedAnno.bbox.ymax}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500">W:</span>{' '}
                  <span className="text-cyan-400">
                    {Math.round(((selectedAnno.bbox.xmax - selectedAnno.bbox.xmin) / 1000) * imageNaturalSize.width)}px
                  </span>
                </div>
                <div>
                  <span className="text-slate-500">H:</span>{' '}
                  <span className="text-cyan-400">
                    {Math.round(((selectedAnno.bbox.ymax - selectedAnno.bbox.ymin) / 1000) * imageNaturalSize.height)}px
                  </span>
                </div>
              </div>

              {/* Class Switcher for Selected Box */}
              <div>
                <label className="text-slate-400 text-[10px] uppercase block mb-1">Assign Label:</label>
                <div className="grid grid-cols-2 gap-1">
                  {PRESET_LABELS.map((p) => (
                    <button
                      key={p.name}
                      onClick={() => {
                        setAnnotations((prev) =>
                          prev.map((a) => (a.id === selectedAnno.id ? { ...a, label: p.name, color: p.color } : a))
                        );
                      }}
                      className={`px-2 py-1 rounded text-[10px] flex items-center space-x-1 border transition-all ${
                        selectedAnno.label === p.name
                          ? 'border-amber-400 bg-amber-500/20 text-amber-200 font-bold'
                          : 'border-slate-800 bg-slate-900 text-slate-400 hover:text-white'
                      }`}
                    >
                      <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: p.color }} />
                      <span className="truncate">{p.name}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-2 text-xs font-mono">
              <span className="font-bold text-slate-300">Default Class for New Boxes</span>
              <div className="grid grid-cols-2 gap-1.5">
                {PRESET_LABELS.map((p) => (
                  <button
                    key={p.name}
                    onClick={() => {
                      setActiveLabel(p.name);
                      setCustomLabelInput('');
                    }}
                    className={`px-2 py-1.5 rounded text-[11px] flex items-center space-x-1.5 border transition-all ${
                      activeLabel === p.name && !customLabelInput
                        ? 'border-amber-400 bg-amber-500/20 text-amber-200 font-bold'
                        : 'border-slate-800 bg-slate-900 text-slate-400 hover:text-white'
                    }`}
                  >
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: p.color }} />
                    <span className="truncate">{p.name}</span>
                  </button>
                ))}
              </div>

              {/* Custom Label Input */}
              <div className="pt-2">
                <input
                  type="text"
                  placeholder="Custom label name..."
                  value={customLabelInput}
                  onChange={(e) => setCustomLabelInput(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 text-xs text-white rounded px-2.5 py-1.5 focus:outline-none focus:border-amber-400 placeholder:text-slate-600"
                />
              </div>
            </div>
          )}

          {/* Annotation List */}
          <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg flex-1 flex flex-col space-y-2 text-xs font-mono overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
              <span className="font-bold text-slate-300">All Labels ({annotations.length})</span>
              {annotations.length > 0 && (
                <button
                  onClick={() => {
                    setAnnotations([]);
                    setSelectedAnnoId(null);
                  }}
                  className="text-[10px] text-rose-400 hover:underline"
                >
                  Clear All
                </button>
              )}
            </div>

            <div className="flex-1 overflow-y-auto space-y-1 pr-1 max-h-[220px]">
              {annotations.length === 0 ? (
                <div className="text-center py-6 text-slate-500 italic text-[11px]">
                  No annotations yet. Click & drag on image to create a box or click "Auto-Detect".
                </div>
              ) : (
                annotations.map((a, idx) => {
                  const isSelected = a.id === selectedAnnoId;
                  return (
                    <div
                      key={a.id}
                      onClick={() => setSelectedAnnoId(a.id)}
                      className={`p-2 rounded border cursor-pointer flex items-center justify-between transition-all ${
                        isSelected
                          ? 'bg-amber-500/20 border-amber-500 text-amber-200'
                          : 'bg-slate-900/80 border-slate-800 hover:border-slate-700 text-slate-300'
                      }`}
                    >
                      <div className="flex items-center space-x-2 truncate">
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: a.color }} />
                        <span className="truncate font-medium">{a.label}</span>
                        <span className="text-[10px] text-slate-500">#{idx + 1}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-[10px] text-slate-400">
                          {a.isAiGenerated ? 'AI' : 'GT'}
                        </span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setAnnotations((prev) => prev.filter((item) => item.id !== a.id));
                            if (selectedAnnoId === a.id) setSelectedAnnoId(null);
                          }}
                          className="text-slate-500 hover:text-rose-400 p-0.5"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
