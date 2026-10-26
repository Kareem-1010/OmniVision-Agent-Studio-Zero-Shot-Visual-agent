/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { InstanceNode } from '../types/index.ts';

interface ThreeSceneViewProps {
  instances: InstanceNode[];
  selectedInstanceId?: string;
  onSelectInstance?: (id: string) => void;
}

export const ThreeSceneView: React.FC<ThreeSceneViewProps> = ({
  instances,
  selectedInstanceId,
  onSelectInstance,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [cameraDistance, setCameraDistance] = useState<number>(8.5);
  const [wireframeMode, setWireframeMode] = useState<boolean>(false);
  const [pointCloudDensity, setPointCloudDensity] = useState<'high' | 'medium'>('high');

  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);
  const boxMeshesGroupRef = useRef<THREE.Group | null>(null);
  const pointCloudMeshRef = useRef<THREE.Points | null>(null);

  // Mouse drag orbit controls
  const isDraggingRef = useRef<boolean>(false);
  const previousMousePositionRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const cameraRotationRef = useRef<{ theta: number; phi: number }>({ theta: 0.35, phi: 0.25 });

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 420;

    // 1. Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090d16);
    scene.fog = new THREE.FogExp2(0x090d16, 0.035);
    sceneRef.current = scene;

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    cameraRef.current = camera;
    updateCameraPosition(camera, cameraRotationRef.current.theta, cameraRotationRef.current.phi, cameraDistance);

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    rendererRef.current = renderer;
    container.replaceChildren(renderer.domElement);

    // 4. Studio Lighting Rig
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0x60a5fa, 2.2);
    keyLight.position.set(5, 10, 7);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0xf43f5e, 1.5);
    rimLight.position.set(-6, -4, -5);
    scene.add(rimLight);

    // 5. Metric Ground Plane Grid (1 meter increments)
    const gridHelper = new THREE.GridHelper(24, 24, 0x38bdf8, 0x1e293b);
    gridHelper.position.y = -1.4;
    scene.add(gridHelper);

    // Camera Frustum Pyramidal Wireframe (representing camera location at origin)
    const frustumGroup = new THREE.Group();
    const frustumGeo = new THREE.ConeGeometry(0.6, 1.2, 4);
    const frustumMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true });
    const frustumMesh = new THREE.Mesh(frustumGeo, frustumMat);
    frustumMesh.rotation.x = Math.PI / 2;
    frustumMesh.position.set(0, 0, 0);
    frustumGroup.add(frustumMesh);

    // Frustum axes helper
    const axesHelper = new THREE.AxesHelper(1.0);
    frustumGroup.add(axesHelper);
    scene.add(frustumGroup);

    // Group for 3D Oriented Bounding Boxes
    const boxGroup = new THREE.Group();
    scene.add(boxGroup);
    boxMeshesGroupRef.current = boxGroup;

    // 6. Animation loop
    const animate = () => {
      animationFrameIdRef.current = requestAnimationFrame(animate);
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
    animate();

    // Resize observer
    const handleResize = () => {
      if (!container || !rendererRef.current || !cameraRef.current) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameIdRef.current) cancelAnimationFrame(animationFrameIdRef.current);
      renderer.dispose();
      container.replaceChildren();
    };
  }, []);

  function updateCameraPosition(cam: THREE.PerspectiveCamera, theta: number, phi: number, dist: number) {
    const x = dist * Math.cos(phi) * Math.sin(theta);
    const y = dist * Math.sin(phi);
    const z = dist * Math.cos(phi) * Math.cos(theta);
    cam.position.set(x, y, z);
    cam.lookAt(0, 0, 4.5); // Look towards objects in front of camera
  }

  // Update Point Cloud and 3D Bounding Boxes when instances change
  useEffect(() => {
    if (!sceneRef.current || !boxMeshesGroupRef.current) return;

    // Clear previous box meshes
    while (boxMeshesGroupRef.current.children.length > 0) {
      const obj = boxMeshesGroupRef.current.children[0];
      boxMeshesGroupRef.current.remove(obj);
      if (obj instanceof THREE.Mesh) {
        obj.geometry.dispose();
        if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
        else obj.material.dispose();
      }
    }

    // Remove old point cloud
    if (pointCloudMeshRef.current) {
      sceneRef.current.remove(pointCloudMeshRef.current);
      pointCloudMeshRef.current.geometry.dispose();
      (pointCloudMeshRef.current.material as THREE.Material).dispose();
      pointCloudMeshRef.current = null;
    }

    // 1. Generate 3D Oriented Bounding Boxes for detected instances
    instances.forEach((inst, idx) => {
      const center = inst.bbox3d?.center || [
        ((inst.bbox2d.xmin + inst.bbox2d.xmax) / 1000 - 0.5) * inst.depth.metricDistanceM * 0.8,
        -((inst.bbox2d.ymin + inst.bbox2d.ymax) / 1000 - 0.5) * inst.depth.metricDistanceM * 0.6,
        inst.depth.metricDistanceM,
      ];

      const dims = inst.bbox3d?.dimensions || [0.9, 1.2, 1.0];
      const isSelected = inst.id === selectedInstanceId;

      // Color palette based on instance type
      let boxColor = 0x3b82f6;
      if (inst.label.includes('hazard') || inst.label.includes('spill')) boxColor = 0xef4444;
      else if (inst.label.includes('cone') || inst.label.includes('cart')) boxColor = 0xf59e0b;
      else if (inst.label.includes('red_car')) boxColor = 0xd946ef;
      else if (inst.label.includes('worker') || inst.label.includes('person')) boxColor = 0x10b981;

      if (isSelected) boxColor = 0xfacc15;

      const boxGeo = new THREE.BoxGeometry(dims[0], dims[1], dims[2]);
      const boxMat = new THREE.MeshStandardMaterial({
        color: boxColor,
        roughness: 0.2,
        metalness: 0.1,
        transparent: true,
        opacity: isSelected ? 0.75 : wireframeMode ? 0.2 : 0.45,
        wireframe: wireframeMode,
      });

      const mesh = new THREE.Mesh(boxGeo, boxMat);
      mesh.position.set(center[0], center[1], center[2]);
      mesh.rotation.y = inst.bbox3d?.yaw || 0;
      mesh.userData = { instanceId: inst.id, label: inst.label };

      // Add wireframe edge highlight
      const wireEdges = new THREE.EdgesGeometry(boxGeo);
      const wireLineMat = new THREE.LineBasicMaterial({
        color: isSelected ? 0xffffff : boxColor,
        linewidth: 2,
      });
      const wireLines = new THREE.LineSegments(wireEdges, wireLineMat);
      mesh.add(wireLines);

      boxMeshesGroupRef.current?.add(mesh);
    });

    // 2. Generate Dense Metric Depth Point Cloud around objects and environment
    const numPoints = pointCloudDensity === 'high' ? 4200 : 2000;
    const positions = new Float32Array(numPoints * 3);
    const colors = new Float32Array(numPoints * 3);

    let pIdx = 0;
    // Ground plane points
    for (let i = 0; i < numPoints * 0.4; i++) {
      const gx = (Math.random() - 0.5) * 12;
      const gz = Math.random() * 14 + 1.0;
      const gy = -1.4 + (Math.random() - 0.5) * 0.05;

      positions[pIdx * 3] = gx;
      positions[pIdx * 3 + 1] = gy;
      positions[pIdx * 3 + 2] = gz;

      // Depth colormap (Turbo style: blue near -> red far)
      const normZ = Math.min(1, gz / 15);
      colors[pIdx * 3] = 0.1 + normZ * 0.8;
      colors[pIdx * 3 + 1] = 0.5 * (1 - normZ);
      colors[pIdx * 3 + 2] = 0.9 * (1 - normZ);
      pIdx++;
    }

    // Points clustered on object surface contours
    instances.forEach((inst) => {
      const center = inst.bbox3d?.center || [0, 0, inst.depth.metricDistanceM];
      const dims = inst.bbox3d?.dimensions || [0.8, 1.2, 0.9];
      const pointsPerObj = Math.floor((numPoints * 0.6) / Math.max(1, instances.length));

      for (let j = 0; j < pointsPerObj; j++) {
        if (pIdx >= numPoints) break;
        const px = center[0] + (Math.random() - 0.5) * dims[0] * 1.1;
        const py = center[1] + (Math.random() - 0.5) * dims[1] * 1.1;
        const pz = center[2] + (Math.random() - 0.5) * dims[2] * 1.1;

        positions[pIdx * 3] = px;
        positions[pIdx * 3 + 1] = py;
        positions[pIdx * 3 + 2] = pz;

        colors[pIdx * 3] = 0.38;
        colors[pIdx * 3 + 1] = 0.74;
        colors[pIdx * 3 + 2] = 0.97;
        pIdx++;
      }
    });

    const pcdGeometry = new THREE.BufferGeometry();
    pcdGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pcdGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const pcdMaterial = new THREE.PointsMaterial({
      size: 0.07,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
    });

    const pointCloud = new THREE.Points(pcdGeometry, pcdMaterial);
    sceneRef.current.add(pointCloud);
    pointCloudMeshRef.current = pointCloud;
  }, [instances, selectedInstanceId, wireframeMode, pointCloudDensity]);

  // Handle Mouse Drag Navigation
  const handleMouseDown = (e: React.MouseEvent) => {
    isDraggingRef.current = true;
    previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingRef.current || !cameraRef.current) return;
    const deltaX = e.clientX - previousMousePositionRef.current.x;
    const deltaY = e.clientY - previousMousePositionRef.current.y;

    cameraRotationRef.current.theta -= deltaX * 0.008;
    cameraRotationRef.current.phi = Math.max(-0.2, Math.min(1.2, cameraRotationRef.current.phi + deltaY * 0.008));

    updateCameraPosition(cameraRef.current, cameraRotationRef.current.theta, cameraRotationRef.current.phi, cameraDistance);
    previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (!cameraRef.current) return;
    const newDist = Math.max(3.0, Math.min(20.0, cameraDistance + e.deltaY * 0.005));
    setCameraDistance(newDist);
    updateCameraPosition(cameraRef.current, cameraRotationRef.current.theta, cameraRotationRef.current.phi, newDist);
  };

  return (
    <div className="relative w-full h-[440px] bg-slate-950 rounded-xl overflow-hidden border border-slate-800 shadow-2xl select-none">
      {/* 3D WebGL Canvas */}
      <div
        ref={mountRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />

      {/* Top Overlay Badges */}
      <div className="absolute top-3 left-3 flex items-center space-x-2 pointer-events-none">
        <span className="px-2.5 py-1 text-xs font-mono font-semibold bg-slate-900/90 text-cyan-400 border border-cyan-800/60 rounded backdrop-blur">
          Depth Anything V2 3D Metric Point Cloud
        </span>
        <span className="px-2 py-0.5 text-[11px] font-mono bg-slate-900/80 text-slate-300 border border-slate-700 rounded">
          Ground-Plane RANSAC (1m Grid)
        </span>
      </div>

      {/* Camera Distance & Density Controls (Collapsible Bottom Dock adhering to Three.js directives) */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center space-x-3 px-4 py-2 bg-slate-900/90 border border-slate-800 rounded-full shadow-lg text-xs font-mono text-slate-300">
        <button
          onClick={() => setWireframeMode(!wireframeMode)}
          className={`px-2.5 py-1 rounded transition-colors ${wireframeMode ? 'bg-cyan-600 text-white' : 'bg-slate-800 hover:bg-slate-700'}`}
        >
          {wireframeMode ? 'Solid View' : 'Wireframe OBB'}
        </button>

        <button
          onClick={() => setPointCloudDensity(pointCloudDensity === 'high' ? 'medium' : 'high')}
          className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 transition-colors"
        >
          Points: {pointCloudDensity.toUpperCase()}
        </button>

        <div className="flex items-center space-x-1 pl-2 border-l border-slate-700">
          <span>Zoom:</span>
          <button
            onClick={() => {
              const d = Math.max(3.0, cameraDistance - 1.5);
              setCameraDistance(d);
              if (cameraRef.current) updateCameraPosition(cameraRef.current, cameraRotationRef.current.theta, cameraRotationRef.current.phi, d);
            }}
            className="w-6 h-6 flex items-center justify-center bg-slate-800 hover:bg-slate-700 rounded"
          >
            +
          </button>
          <button
            onClick={() => {
              const d = Math.min(20.0, cameraDistance + 1.5);
              setCameraDistance(d);
              if (cameraRef.current) updateCameraPosition(cameraRef.current, cameraRotationRef.current.theta, cameraRotationRef.current.phi, d);
            }}
            className="w-6 h-6 flex items-center justify-center bg-slate-800 hover:bg-slate-700 rounded"
          >
            -
          </button>
        </div>

        <button
          onClick={() => {
            cameraRotationRef.current = { theta: 0.35, phi: 0.25 };
            setCameraDistance(8.5);
            if (cameraRef.current) updateCameraPosition(cameraRef.current, 0.35, 0.25, 8.5);
          }}
          className="px-2.5 py-1 text-slate-400 hover:text-white rounded bg-slate-800"
        >
          Reset Camera
        </button>
      </div>

      {/* Right-hand side 3D Instance Chips */}
      <div className="absolute top-3 right-3 flex flex-col space-y-1.5 max-h-[300px] overflow-y-auto pointer-events-auto">
        {instances.slice(0, 5).map((inst) => (
          <button
            key={inst.id}
            onClick={() => onSelectInstance && onSelectInstance(inst.id)}
            className={`px-2.5 py-1 text-left text-xs font-mono rounded border transition-all ${
              inst.id === selectedInstanceId
                ? 'bg-amber-500/20 text-amber-300 border-amber-400'
                : 'bg-slate-900/80 text-slate-300 border-slate-800 hover:border-slate-700'
            }`}
          >
            <div className="font-semibold">{inst.label}</div>
            <div className="text-[10px] text-slate-400">
              Depth: {inst.depth.metricDistanceM}m | 3D: [{inst.bbox3d?.center.map((v) => v.toFixed(1)).join(', ')}]
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};
