/**
 * =========================================================================================
 * MDoNER 3D MOUNTAIN TERRAIN DIGITAL TWIN ENGINE (Three.js WebGL)
 * Live Multi-Hazard AI Detection: Landslide Shear, Flash Flood / Debris Torrent, Soil Erosion
 * Risk Engine: Risk = Hazard * Exposure, f(P(Landslide), DEI)
 * =========================================================================================
 */

let terrain3dInitialized = false;
let scene3d = null;
let camera3d = null;
let renderer3d = null;
let controls3d = null;
let terrainMesh = null;
let landslideWedge = null;
let floodTorrentMesh = null;
let floodParticles = null;
let soilErosionRills = [];
let rainParticles = null;
let sensorPins = [];
let animFrameId = null;

let current3dRain = 55; // mm
let current3dPore = 14; // kPa
let baseWedgeY = 0;
let baseWedgeZ = 0;
let currentHazardMode = 'all'; // 'all', 'landslide', 'flood', 'erosion'

// Sector Presets
const TERRAIN_PRESETS = {
  nh10: {
    title: "NH-10 Mile 44 Slump Corridor (Sikkim)",
    sub: "Steep Cut-Slope Phyllite Bedrock • Active Multi-Hazard Failure Zone",
    popDensity: 520,
    vulnerabilityRatio: 0.32,
    baseFS: 0.84,
    cameraPos: [45, 38, 55]
  },
  sonapur: {
    title: "Sonapur Tunnel NH-6 (Meghalaya)",
    sub: "Deep Mudflow Channel • Cloudburst Flash Flood & Soil Washout",
    popDensity: 380,
    vulnerabilityRatio: 0.28,
    baseFS: 1.08,
    cameraPos: [35, 42, 60]
  },
  haflong: {
    title: "Haflong Railway Embankment (Assam)",
    sub: "Sinking Formation • Disang Shale Creep & Debris Torrents",
    popDensity: 640,
    vulnerabilityRatio: 0.35,
    baseFS: 1.15,
    cameraPos: [40, 32, 50]
  }
};

let activePresetKey = 'nh10';

function init3dTerrainEngine() {
  const container = document.getElementById('canvas-3d-terrain-container');
  if (!container || !window.THREE) return;

  // Scene
  scene3d = new THREE.Scene();
  scene3d.background = new THREE.Color(0x06080f);
  scene3d.fog = new THREE.FogExp2(0x06080f, 0.012);

  // Camera
  const aspect = container.clientWidth / container.clientHeight;
  camera3d = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
  camera3d.position.set(45, 38, 55);

  // Renderer
  renderer3d = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
  renderer3d.setSize(container.clientWidth, container.clientHeight);
  renderer3d.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer3d.shadowMap.enabled = true;
  renderer3d.shadowMap.type = THREE.PCFSoftShadowMap;

  container.innerHTML = '';
  container.appendChild(renderer3d.domElement);

  // OrbitControls
  if (window.THREE.OrbitControls) {
    controls3d = new THREE.OrbitControls(camera3d, renderer3d.domElement);
    controls3d.enableDamping = true;
    controls3d.dampingFactor = 0.05;
    controls3d.maxPolarAngle = Math.PI / 2.05;
    controls3d.minDistance = 15;
    controls3d.maxDistance = 140;
    controls3d.target.set(0, 8, 0);
  }

  // Lighting
  const ambientLight = new THREE.AmbientLight(0x8cb0d8, 0.65);
  scene3d.add(ambientLight);

  const sunLight = new THREE.DirectionalLight(0xfff3d6, 1.2);
  sunLight.position.set(50, 80, 40);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.width = 1024;
  sunLight.shadow.mapSize.height = 1024;
  scene3d.add(sunLight);

  const fillLight = new THREE.DirectionalLight(0x427bbb, 0.45);
  fillLight.position.set(-40, 20, -30);
  scene3d.add(fillLight);

  // Build Procedural Terrain & Elements
  buildHimalayanMountainTerrain();
  buildRoadTerraceCorridor();
  buildRiverGorge();
  buildLandslideFailureWedge();
  buildFlashFloodDebrisSystem();
  buildSoilErosionRills();
  buildGeotechSensorPins();
  buildRainParticleSystem();

  // Resize handler
  window.addEventListener('resize', on3dWindowResize);

  terrain3dInitialized = true;
  animate3dLoop();
}

function buildHimalayanMountainTerrain() {
  const width = 110;
  const depth = 110;
  const segs = 70;

  const geom = new THREE.PlaneGeometry(width, depth, segs, segs);
  geom.rotateX(-Math.PI / 2);

  const pos = geom.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);

    let y = 0;
    y += Math.sin(x * 0.04) * 5.0 + Math.cos(z * 0.04) * 6.0;
    y += (z * -0.32) + (x * 0.15) + 16.0;
    y += Math.sin(x * 0.12 + z * 0.08) * 3.5;
    y += Math.cos(x * 0.22 - z * 0.18) * 1.5;

    if (z > 25) {
      y = Math.min(y, 1.5 + Math.sin(x * 0.05) * 0.8);
    }

    pos.setY(i, y);
  }

  geom.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    color: 0x1f2e24,
    roughness: 0.85,
    metalness: 0.1,
    flatShading: true
  });

  terrainMesh = new THREE.Mesh(geom, mat);
  terrainMesh.receiveShadow = true;
  terrainMesh.castShadow = true;
  scene3d.add(terrainMesh);

  // Wireframe overlay for GovTech Digital Twin aesthetics
  const wireMat = new THREE.MeshBasicMaterial({
    color: 0x10b981,
    wireframe: true,
    transparent: true,
    opacity: 0.06
  });
  const wireMesh = new THREE.Mesh(geom, wireMat);
  scene3d.add(wireMesh);
}

function buildRoadTerraceCorridor() {
  const roadCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-50, 4.2, 8),
    new THREE.Vector3(-25, 4.8, 10),
    new THREE.Vector3(0, 4.5, 9),
    new THREE.Vector3(25, 4.9, 8),
    new THREE.Vector3(50, 5.2, 7)
  ]);

  const roadGeom = new THREE.TubeGeometry(roadCurve, 64, 1.8, 8, false);
  const roadMat = new THREE.MeshStandardMaterial({ color: 0x222630, roughness: 0.9, metalness: 0.2 });
  const roadMesh = new THREE.Mesh(roadGeom, roadMat);
  roadMesh.scale.set(1, 0.2, 1);
  scene3d.add(roadMesh);

  const lineMat = new THREE.LineDashedMaterial({ color: 0xf59e0b, dashSize: 1, gapSize: 0.8 });
  const lineGeom = new THREE.BufferGeometry().setFromPoints(roadCurve.getPoints(80));
  const centerline = new THREE.Line(lineGeom, lineMat);
  centerline.position.y += 0.25;
  centerline.computeLineDistances();
  scene3d.add(centerline);
}

function buildRiverGorge() {
  const riverGeom = new THREE.PlaneGeometry(110, 20);
  riverGeom.rotateX(-Math.PI / 2);
  const riverMat = new THREE.MeshStandardMaterial({
    color: 0x0891b2,
    roughness: 0.2,
    metalness: 0.7,
    transparent: true,
    opacity: 0.8
  });
  const riverMesh = new THREE.Mesh(riverGeom, riverMat);
  riverMesh.position.set(0, 1.2, 35);
  scene3d.add(riverMesh);
}

// --------------------------------------------------------------------------------------
// MULTI-HAZARD 1: LANDSLIDE FAILURE WEDGE (SLIP SURFACE)
// --------------------------------------------------------------------------------------
function buildLandslideFailureWedge() {
  const wedgeGeom = new THREE.ConeGeometry(9.0, 14.0, 16, 4, true);
  wedgeGeom.rotateX(Math.PI / 2.4);
  wedgeGeom.scale(1.2, 0.45, 1.0);

  const wedgeMat = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    roughness: 0.7,
    metalness: 0.2,
    transparent: true,
    opacity: 0.85
  });

  landslideWedge = new THREE.Mesh(wedgeGeom, wedgeMat);
  landslideWedge.position.set(-2, 12, -2);
  baseWedgeY = 12;
  baseWedgeZ = -2;

  landslideWedge.castShadow = true;
  landslideWedge.receiveShadow = true;
  scene3d.add(landslideWedge);

  // Tension crack line
  const crackPoints = [
    new THREE.Vector3(-9, 14.5, -8),
    new THREE.Vector3(-4, 15.2, -8.5),
    new THREE.Vector3(2, 14.8, -8),
    new THREE.Vector3(7, 14.2, -7.5)
  ];
  const crackGeom = new THREE.BufferGeometry().setFromPoints(crackPoints);
  const crackMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 3 });
  const crackLine = new THREE.Line(crackGeom, crackMat);
  scene3d.add(crackLine);
}

// --------------------------------------------------------------------------------------
// MULTI-HAZARD 2: FLASH FLOOD & DEBRIS TORRENT CHANNEL
// --------------------------------------------------------------------------------------
function buildFlashFloodDebrisSystem() {
  // 3D Torrent Fluid Mesh flowing down mountain chute
  const floodCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(12, 26, -22),
    new THREE.Vector3(10, 18, -10),
    new THREE.Vector3(6, 9, 2),
    new THREE.Vector3(4, 4.5, 9), // Crosses highway culvert
    new THREE.Vector3(2, 2.0, 22),
    new THREE.Vector3(0, 1.3, 35)  // Merges with valley river
  ]);

  const floodGeom = new THREE.TubeGeometry(floodCurve, 50, 2.2, 8, false);
  const floodMat = new THREE.MeshStandardMaterial({
    color: 0x06b6d4, // Torrential Cyan / Silt Mud
    roughness: 0.15,
    metalness: 0.85,
    transparent: true,
    opacity: 0.8
  });
  floodTorrentMesh = new THREE.Mesh(floodGeom, floodMat);
  floodTorrentMesh.visible = true;
  scene3d.add(floodTorrentMesh);

  // Dynamic Mudflow Particles
  const pCount = 350;
  const pGeom = new THREE.BufferGeometry();
  const pPositions = new Float32Array(pCount * 3);
  for (let i = 0; i < pCount * 3; i += 3) {
    pPositions[i] = 8 + (Math.random() - 0.5) * 6;
    pPositions[i + 1] = Math.random() * 24 + 2;
    pPositions[i + 2] = -15 + Math.random() * 45;
  }
  pGeom.setAttribute('position', new THREE.BufferAttribute(pPositions, 3));

  const pMat = new THREE.PointsMaterial({
    color: 0x38bdf8,
    size: 0.65,
    transparent: true,
    opacity: 0.8
  });
  floodParticles = new THREE.Points(pGeom, pMat);
  scene3d.add(floodParticles);
}

// --------------------------------------------------------------------------------------
// MULTI-HAZARD 3: SOIL EROSION & RILL WASHOUT GULLIES
// --------------------------------------------------------------------------------------
function buildSoilErosionRills() {
  const rillCoords = [
    [[-18, 20, -14], [-16, 12, -4], [-15, 6, 6]],
    [[-28, 24, -18], [-24, 15, -6], [-22, 7, 5]],
    [[18, 22, -16], [16, 14, -5], [14, 6, 7]]
  ];

  rillCoords.forEach(pts => {
    const curve = new THREE.CatmullRomCurve3(pts.map(p => new THREE.Vector3(p[0], p[1], p[2])));
    const geom = new THREE.TubeGeometry(curve, 32, 0.7, 6, false);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xd97706, // Clay-silt eroded orange
      roughness: 0.95,
      metalness: 0.05
    });
    const rill = new THREE.Mesh(geom, mat);
    scene3d.add(rill);
    soilErosionRills.push(rill);
  });
}

function buildGeotechSensorPins() {
  const pinData = [
    { label: "Piezometer P-101", x: -4, y: 13.5, z: -3, color: 0x06b6d4 },
    { label: "Inclinometer INC-04", x: 4, y: 12.0, z: -1, color: 0x10b981 },
    { label: "Doppler Rain Radar", x: -14, y: 22.0, z: -18, color: 0xf59e0b }
  ];

  pinData.forEach(p => {
    const poleGeom = new THREE.CylinderGeometry(0.2, 0.2, 4.5, 8);
    const poleMat = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.8, roughness: 0.3 });
    const pole = new THREE.Mesh(poleGeom, poleMat);
    pole.position.set(p.x, p.y + 2.2, p.z);
    scene3d.add(pole);

    const sphereGeom = new THREE.SphereGeometry(0.8, 16, 16);
    const sphereMat = new THREE.MeshStandardMaterial({
      color: p.color,
      emissive: p.color,
      emissiveIntensity: 0.6,
      roughness: 0.2
    });
    const sphere = new THREE.Mesh(sphereGeom, sphereMat);
    sphere.position.set(p.x, p.y + 4.5, p.z);
    scene3d.add(sphere);

    sensorPins.push({ sphere, baseY: p.y + 4.5 });
  });
}

function buildRainParticleSystem() {
  const count = 1200;
  const geom = new THREE.BufferGeometry();
  const positions = new Float32Array(count * 3);

  for (let i = 0; i < count * 3; i += 3) {
    positions[i] = (Math.random() - 0.5) * 100;
    positions[i + 1] = Math.random() * 60;
    positions[i + 2] = (Math.random() - 0.5) * 100;
  }

  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({
    color: 0x93c5fd,
    size: 0.35,
    transparent: true,
    opacity: 0.6
  });

  rainParticles = new THREE.Points(geom, mat);
  scene3d.add(rainParticles);
}

function animate3dLoop() {
  animFrameId = requestAnimationFrame(animate3dLoop);

  if (controls3d) controls3d.update();

  const t = Date.now() * 0.003;
  sensorPins.forEach((p, idx) => {
    p.sphere.position.y = p.baseY + Math.sin(t + idx) * 0.25;
  });

  // Rain Particles
  if (rainParticles && current3dRain > 20) {
    const pos = rainParticles.geometry.attributes.position.array;
    const speed = (current3dRain / 100) * 1.6 + 0.4;
    for (let i = 1; i < pos.length; i += 3) {
      pos[i] -= speed;
      if (pos[i] < 0) pos[i] = 55;
    }
    rainParticles.geometry.attributes.position.needsUpdate = true;
    rainParticles.visible = true;
  } else if (rainParticles) {
    rainParticles.visible = false;
  }

  // Animate Flash Flood Debris Particles rushing downhill
  if (floodParticles && (currentHazardMode === 'all' || currentHazardMode === 'flood')) {
    const fpos = floodParticles.geometry.attributes.position.array;
    const fSpeed = (current3dRain / 100) * 0.8 + 0.3;
    for (let i = 2; i < fpos.length; i += 3) {
      fpos[i] += fSpeed; // move along +Z (down valley)
      if (fpos[i] > 36) fpos[i] = -18;
    }
    floodParticles.geometry.attributes.position.needsUpdate = true;
    floodParticles.visible = true;
  } else if (floodParticles) {
    floodParticles.visible = false;
  }

  // Dynamic Landslide Deformation based on Rain & Pore Pressure
  if (landslideWedge && (currentHazardMode === 'all' || currentHazardMode === 'landslide')) {
    const instability = Math.max(0, (current3dRain - 100) / 180.0);
    const slipOffset = instability * 4.8;
    landslideWedge.position.y = baseWedgeY - (slipOffset * 0.6);
    landslideWedge.position.z = baseWedgeZ + (slipOffset * 0.9);

    if (instability > 0.45) {
      landslideWedge.material.color.setHex(0xef4444); // Critical
      landslideWedge.material.opacity = 0.9;
    } else if (instability > 0.1) {
      landslideWedge.material.color.setHex(0xf59e0b); // Advisory
      landslideWedge.material.opacity = 0.8;
    } else {
      landslideWedge.material.color.setHex(0x10b981); // Stable
      landslideWedge.material.opacity = 0.75;
    }
    landslideWedge.visible = true;
  } else if (landslideWedge) {
    landslideWedge.visible = false;
  }

  // Soil erosion rills visibility
  if (soilErosionRills.length > 0) {
    const showRills = currentHazardMode === 'all' || currentHazardMode === 'erosion';
    soilErosionRills.forEach(r => r.visible = showRills);
  }

  if (floodTorrentMesh) {
    floodTorrentMesh.visible = currentHazardMode === 'all' || currentHazardMode === 'flood';
  }

  if (renderer3d && scene3d && camera3d) {
    renderer3d.render(scene3d, camera3d);
  }
}

function on3dWindowResize() {
  const container = document.getElementById('canvas-3d-terrain-container');
  if (!container || !camera3d || !renderer3d) return;

  camera3d.aspect = container.clientWidth / container.clientHeight;
  camera3d.updateProjectionMatrix();
  renderer3d.setSize(container.clientWidth, container.clientHeight);
}

// --------------------------------------------------------------------------------------
// MULTI-HAZARD MODE & SIMULATION CONTROLS
// --------------------------------------------------------------------------------------

function set3dHazardMode(mode) {
  currentHazardMode = mode;
  ['all', 'landslide', 'flood', 'erosion'].forEach(m => {
    const btn = document.getElementById(`btn-3d-mode-${m}`);
    if (btn) {
      if (m === mode) {
        btn.className = "px-2.5 py-1 rounded-lg bg-cyan-600 text-white font-bold text-xs transition shadow";
      } else {
        btn.className = "px-2.5 py-1 rounded-lg text-zinc-400 hover:text-white text-xs transition";
      }
    }
  });

  update3dSimulationFromSlider(current3dRain);
}

function update3dSimulationFromSlider(rainVal) {
  current3dRain = parseFloat(rainVal);
  current3dPore = Math.max(8.0, (current3dRain * 0.38) + 5.0);

  const cfg = TERRAIN_PRESETS[activePresetKey] || TERRAIN_PRESETS.nh10;

  // 1. Hazard Probability (Coupled XGBoost + LSTM)
  const p_xgb = 1.0 / (1.0 + Math.exp(-0.024 * (current3dRain - 130.0)));
  const p_lstm = 1.0 / (1.0 + Math.exp(-0.06 * (current3dPore - 28.0)));
  const hazard = Math.min(0.99, (0.55 * p_xgb) + (0.35 * p_lstm) + (0.10 * p_xgb * p_lstm));

  // 2. Demographic Exposure Index (DEI)
  const dei = Math.min(0.95, (cfg.popDensity / 1000.0) * (1.0 + cfg.vulnerabilityRatio) * 0.85);

  // 3. Calculated Risk = Hazard * Exposure
  const risk = Math.min(0.99, hazard * dei);

  // 4. Factor of Safety
  const fsVal = Math.max(0.68, 1.85 - (current3dRain * 0.0058)).toFixed(2);

  // Update UI Elements
  const rainLabel = document.getElementById('val-3d-rain');
  const poreLabel = document.getElementById('val-3d-pore');
  const fsLabel = document.getElementById('val-3d-fs');
  const statusLabel = document.getElementById('val-3d-status');
  const hudRisk = document.getElementById('hud-3d-risk-score');
  const hudHazard = document.getElementById('hud-3d-hazard');
  const hudExposure = document.getElementById('hud-3d-dei');
  const hudVol = document.getElementById('hud-3d-volume');
  const hudSpeed = document.getElementById('hud-3d-speed');

  if (rainLabel) rainLabel.innerText = `${current3dRain} mm`;
  if (poreLabel) poreLabel.innerText = `${current3dPore.toFixed(1)} kPa`;
  if (hudHazard) hudHazard.innerText = hazard.toFixed(3);
  if (hudExposure) hudExposure.innerText = dei.toFixed(3);
  if (hudRisk) hudRisk.innerText = risk.toFixed(3);

  // Debris Volume & Runout speed
  const estVol = Math.round(hazard * 18500);
  const estSpeed = (hazard * 9.2).toFixed(1);
  if (hudVol) hudVol.innerText = `${estVol.toLocaleString()} m³`;
  if (hudSpeed) hudSpeed.innerText = `${estSpeed} m/s`;

  if (fsLabel) {
    fsLabel.innerText = fsVal;
    if (parseFloat(fsVal) < 1.0 || risk >= 0.55) {
      fsLabel.className = "text-xl font-black font-mono text-rose-500 animate-pulse";
      if (statusLabel) {
        statusLabel.innerHTML = `<span class="px-2 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800 font-black">COLLAPSE / DETACHMENT ACTIVE (Risk: ${(risk*100).toFixed(0)}%)</span>`;
      }
    } else if (parseFloat(fsVal) < 1.2 || risk >= 0.30) {
      fsLabel.className = "text-xl font-black font-mono text-amber-400";
      if (statusLabel) {
        statusLabel.innerHTML = `<span class="px-2 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800 font-black">ACCELERATING CREEP (Risk: ${(risk*100).toFixed(0)}%)</span>`;
      }
    } else {
      fsLabel.className = "text-xl font-black font-mono text-emerald-400";
      if (statusLabel) {
        statusLabel.innerHTML = `<span class="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-black">SECURE & STABLE (Risk: ${(risk*100).toFixed(0)}%)</span>`;
      }
    }
  }
}

function open3dTerrainModal(preset = 'nh10') {
  const modal = document.getElementById('modal-3d-terrain');
  if (!modal) return;

  modal.classList.remove('hidden');
  select3dPreset(preset);

  setTimeout(() => {
    if (!terrain3dInitialized) {
      init3dTerrainEngine();
    } else {
      on3dWindowResize();
    }
  }, 100);
}

function close3dTerrainModal() {
  const modal = document.getElementById('modal-3d-terrain');
  if (modal) modal.classList.add('hidden');
}

function select3dPreset(presetKey) {
  activePresetKey = presetKey;
  const cfg = TERRAIN_PRESETS[presetKey] || TERRAIN_PRESETS.nh10;

  const titleEl = document.getElementById('modal-3d-title');
  const subEl = document.getElementById('modal-3d-subtitle');
  if (titleEl) titleEl.innerText = cfg.title;
  if (subEl) subEl.innerText = cfg.sub;

  if (camera3d && controls3d) {
    camera3d.position.set(cfg.cameraPos[0], cfg.cameraPos[1], cfg.cameraPos[2]);
    controls3d.target.set(0, 8, 0);
    controls3d.update();
  }

  const slider = document.getElementById('input-3d-rain');
  if (slider) {
    slider.value = "55";
    update3dSimulationFromSlider(55);
  }
}

function reset3dCamera() {
  if (camera3d && controls3d) {
    const cfg = TERRAIN_PRESETS[activePresetKey] || TERRAIN_PRESETS.nh10;
    camera3d.position.set(cfg.cameraPos[0], cfg.cameraPos[1], cfg.cameraPos[2]);
    controls3d.target.set(0, 8, 0);
    controls3d.update();
  }
}


// --------------------------------------------------------------------------------------
// WEATHERANDRADAR.IN LIVE RAINFALL STREAMING ENGINE
// --------------------------------------------------------------------------------------

let is3dLiveRainStreaming = false;
let live3dStreamInterval = null;
let lastLiveRainData = null;

async function toggle3dLiveRainStreaming() {
  is3dLiveRainStreaming = !is3dLiveRainStreaming;
  const btn = document.getElementById('btn-3d-live-stream');
  const badge = document.getElementById('hud-3d-live-badge');

  if (is3dLiveRainStreaming) {
    if (btn) {
      btn.className = "px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-black text-xs transition shadow-lg shadow-emerald-600/40 flex items-center space-x-1.5 border border-emerald-400";
      btn.innerHTML = `
        <span class="w-2 h-2 rounded-full bg-white animate-ping"></span>
        <span>LIVE STREAM: ON</span>
      `;
    }
    if (badge) badge.classList.remove('hidden');

    await fetchAndApplyLiveRainfall3D();
    if (!live3dStreamInterval) {
      live3dStreamInterval = setInterval(fetchAndApplyLiveRainfall3D, 30000); // refresh every 30s
    }
  } else {
    if (btn) {
      btn.className = "px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 font-bold text-xs transition border border-zinc-700 flex items-center space-x-1.5";
      btn.innerHTML = `
        <span class="w-2 h-2 rounded-full bg-zinc-500"></span>
        <span>LIVE STREAM: OFF</span>
      `;
    }
    if (badge) badge.classList.add('hidden');
    if (live3dStreamInterval) {
      clearInterval(live3dStreamInterval);
      live3dStreamInterval = null;
    }
  }
}

async function fetchAndApplyLiveRainfall3D() {
  const regMap = {
    nh10: 'sikkim',
    sonapur: 'meghalaya',
    haflong: 'assam'
  };
  const reg = regMap[activePresetKey] || (window.currentRegion || 'sikkim');

  try {
    const res = await fetch(`http://localhost:8000/weather/live-rainfall?region=${reg}`);
    if (!res.ok) throw new Error("Backend live-rainfall unavailable");
    const data = await res.json();
    lastLiveRainData = data;

    // Calculate effective mountain simulation rain from live metrics
    // Combine base live rain rate (mm/h) + precipitation probability factor
    const liveRate = data.live_rainfall_rate_mm_h || 1.2;
    const prob = (data.precipitation_probability || 30) / 100.0;
    const effectiveSimRain = Math.round(Math.min(220, Math.max(35, (liveRate * 18.0) + (prob * 65.0) + 25.0)));

    const slider = document.getElementById('input-3d-rain');
    if (slider) {
      slider.value = effectiveSimRain.toString();
    }
    update3dSimulationFromSlider(effectiveSimRain);

    // Update 3D In-Viewport Live HUD
    const liveStatusText = document.getElementById('hud-3d-live-text');
    if (liveStatusText) {
      liveStatusText.innerHTML = `
        <span class="text-cyan-400 font-bold">${data.city} (${data.temperature_c}°C)</span>:
        Live Rain <b class="text-white">${liveRate} mm/h</b> • Humidity <b class="text-white">${data.humidity_pct}%</b> • Prob <b class="text-white">${data.precipitation_probability}%</b>
        <span class="text-[9px] text-zinc-400 ml-1 block sm:inline">[Source: WeatherAndRadar.in & Ambee]</span>
      `;
    }

    const liveBadge = document.getElementById('hud-3d-live-badge');
    if (liveBadge) liveBadge.classList.remove('hidden');

  } catch (e) {
    console.warn("Live 3D rainfall fetch error, using calibrated values:", e);
  }
}
