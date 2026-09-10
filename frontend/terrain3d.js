/**
 * =========================================================================================
 * MDoNER 3D MOUNTAIN TERRAIN DIGITAL TWIN & GOOGLE EARTH 3D TOPOGRAPHIC ENGINE
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
let currentViewMode3D = 'webgl'; // 'webgl' or 'google_earth'

// Cinematic Drone Orbit Variables
let isDroneOrbiting = false;
let droneOrbitAngle = 0;

// Leaflet satellite mini-map instance for Google Earth 3D view
let gearthSatelliteMap = null;
let gearthMarkerGroup = null;

// =========================================================================================
// 8-STATE NORTHEAST INDIA DISASTER CORRIDOR PRESETS
// =========================================================================================
const TERRAIN_PRESETS = {
  nh10: {
    key: "nh10",
    state: "Sikkim",
    title: "NH-10 Singtam Gorge & Mile 44 Slump (Sikkim)",
    sub: "Steep Cut-Slope Phyllite Bedrock • Active Multi-Hazard Failure Zone",
    popDensity: 520,
    vulnerabilityRatio: 0.32,
    baseFS: 0.84,
    cameraPos: [45, 38, 55],
    lat: 27.2345,
    lon: 88.4987,
    alt_m: 720,
    heading: 35,
    tilt: 68,
    range: 1800,
    bedrock: "Daling Group Phyllite & Chlorite-Muscovite Schist",
    foliation: "Dip 42° towards SW (Teesta River Gorge)",
    insar_rate: "-14.2 mm/yr (LOS Creep)",
    lifeline: "NH-10 Sevoke-Gangtok (Arterial Single Point of Failure)",
    recommendedMitigation: "Bored micropile walls, sub-horizontal drains, automated acoustic crack sensors",
    gearth_url: "https://earth.google.com/web/@27.2345,88.4987,720a,1800d,35y,68h,0r"
  },
  sonapur: {
    key: "sonapur",
    state: "Meghalaya",
    title: "Sonapur Tunnel NH-6 Hazard Sector (Meghalaya)",
    sub: "Deep Mudflow Channel • Cloudburst Flash Flood & Karst Washout",
    popDensity: 380,
    vulnerabilityRatio: 0.28,
    baseFS: 1.08,
    cameraPos: [35, 42, 60],
    lat: 25.1128,
    lon: 92.3619,
    alt_m: 430,
    heading: 70,
    tilt: 65,
    range: 2200,
    bedrock: "Shella Formation Sandstone & Eocene Limestone Karst",
    foliation: "Dip 28° towards SE (Lukha River Valley)",
    insar_rate: "-11.8 mm/yr (Karst Creep)",
    lifeline: "NH-6 Silchar-Shillong (Tri-State Southern Lifeline)",
    recommendedMitigation: "Rockfall catch net fences, culvert debris deflector, subsurface drainage adits",
    gearth_url: "https://earth.google.com/web/@25.1128,92.3619,430a,2200d,70y,65h,0r"
  },
  haflong: {
    key: "haflong",
    state: "Assam",
    title: "Haflong Hill Railway Sinking Section (Assam)",
    sub: "Sinking Formation • Disang Shale Creep & Debris Torrents",
    popDensity: 640,
    vulnerabilityRatio: 0.35,
    baseFS: 1.15,
    cameraPos: [40, 32, 50],
    lat: 25.1683,
    lon: 93.0182,
    alt_m: 680,
    heading: 120,
    tilt: 62,
    range: 2400,
    bedrock: "Disang Formation Black Shales & Siltstones",
    foliation: "Dip 54° towards NE (Jatinga Valley Fault)",
    insar_rate: "-18.5 mm/yr (Deep Slump)",
    lifeline: "Lumding-Badarpur Broad Gauge Hill Section",
    recommendedMitigation: "Reinforced soil berms, geogrid stabilization, deep horizontal drainage boreholes",
    gearth_url: "https://earth.google.com/web/@25.1683,93.0182,680a,2400d,120y,62h,0r"
  },
  sela: {
    key: "sela",
    state: "Arunachal Pradesh",
    title: "Sela Pass High-Altitude Corridor (Arunachal Pradesh)",
    sub: "Permafrost Freeze-Thaw Rockfall • Balipara-Charduar-Tawang Highway",
    popDensity: 140,
    vulnerabilityRatio: 0.22,
    baseFS: 1.25,
    cameraPos: [48, 44, 52],
    lat: 27.5050,
    lon: 92.1039,
    alt_m: 4170,
    heading: 15,
    tilt: 70,
    range: 3200,
    bedrock: "Central Crystallines High-Grade Gneiss & Migmatites",
    foliation: "Joint set striking NW-SE dipping 68° towards highway",
    insar_rate: "-6.4 mm/yr (Cryogenic Creep)",
    lifeline: "BCT Strategic Highway (Tawang Border Lifeline)",
    recommendedMitigation: "Rock bolt stitching, high-tensile wire mesh drapery, snow avalanche barriers",
    gearth_url: "https://earth.google.com/web/@27.5050,92.1039,4170a,3200d,15y,70h,0r"
  },
  noney: {
    key: "noney",
    state: "Manipur",
    title: "Noney Railway Pier 164 Escarpment (Manipur)",
    sub: "Deep-Seated Rotational Shear Zone • Tupul River Disaster Sector",
    popDensity: 290,
    vulnerabilityRatio: 0.31,
    baseFS: 0.92,
    cameraPos: [38, 36, 58],
    lat: 24.8167,
    lon: 93.5975,
    alt_m: 450,
    heading: 85,
    tilt: 66,
    range: 2100,
    bedrock: "Surma Group Shale & Interbedded Soft Mudstones",
    foliation: "Dip 38° towards Ijei River gorge",
    insar_rate: "-22.1 mm/yr (Rapid Progressive Failure)",
    lifeline: "Jiribam-Imphal Strategic Railway Line",
    recommendedMitigation: "Concrete pile array, toe counter-weight embankment, slope reprofiling",
    gearth_url: "https://earth.google.com/web/@24.8167,93.5975,450a,2100d,85y,66h,0r"
  },
  hunthar: {
    key: "hunthar",
    state: "Mizoram",
    title: "Hunthar Sinking Ridge & Aizawl Escarpment (Mizoram)",
    sub: "Urban Hill-Town Slope Creep • Anticlinal Valley Sinking Formation",
    popDensity: 880,
    vulnerabilityRatio: 0.42,
    baseFS: 0.88,
    cameraPos: [42, 35, 54],
    lat: 23.7431,
    lon: 92.7078,
    alt_m: 910,
    heading: 45,
    tilt: 65,
    range: 2400,
    bedrock: "Bhuban Formation Alternating Sandstone & Fragile Siltstone",
    foliation: "Dip 32° towards Chite Lui Stream",
    insar_rate: "-16.7 mm/yr (Urban Foundation Settling)",
    lifeline: "Aizawl-Lengpui Airport Access Highway",
    recommendedMitigation: "Surface stormwater drainage canalization, gabion check dams, building load caps",
    gearth_url: "https://earth.google.com/web/@23.7431,92.7078,910a,2400d,45y,65h,0r"
  },
  paglapahar: {
    key: "paglapahar",
    state: "Nagaland",
    title: "NH-29 Paglapahar Sinking Stretch (Nagaland)",
    sub: "Chathe River Undercutting & Monsoon Debris Torrent Washout",
    popDensity: 460,
    vulnerabilityRatio: 0.34,
    baseFS: 0.95,
    cameraPos: [44, 38, 56],
    lat: 25.7511,
    lon: 93.7411,
    alt_m: 290,
    heading: 100,
    tilt: 62,
    range: 1900,
    bedrock: "Tipam Sandstone & Overlying Unconsolidated Colluvium",
    foliation: "Dip 45° towards Chathe River Bed",
    insar_rate: "-19.3 mm/yr (Toe Erosion Subsidence)",
    lifeline: "NH-29 Dimapur-Kohima Main Supply Route",
    recommendedMitigation: "Rip-rap river toe armor, shotcrete with wire mesh, bio-engineering vetiver roots",
    gearth_url: "https://earth.google.com/web/@25.7511,93.7411,290a,1900d,100y,62h,0r"
  },
  baramura: {
    key: "baramura",
    state: "Tripura",
    title: "NH-8 Baramura Hill Cut Ridge (Tripura)",
    sub: "Saturated Soft Neogene Sandstone Slope Failure & Gully Chute",
    popDensity: 310,
    vulnerabilityRatio: 0.25,
    baseFS: 1.12,
    cameraPos: [36, 30, 48],
    lat: 23.8315,
    lon: 91.4589,
    alt_m: 210,
    heading: 50,
    tilt: 58,
    range: 1700,
    bedrock: "Bokabil Formation Soft Friable Sandstone & Mottled Clay",
    foliation: "Dip 22° sub-horizontal bed",
    insar_rate: "-8.9 mm/yr (Monsoon Slump)",
    lifeline: "NH-8 Agartala-Assam Interstate Trunk Line",
    recommendedMitigation: "Chute drains, terraced slope benching, hydro-seeding native grass cover",
    gearth_url: "https://earth.google.com/web/@23.8315,91.4589,210a,1700d,50y,58h,0r"
  }
};

let activePresetKey = 'nh10';

// =========================================================================================
// THREE.JS 3D WEBGL ENGINE INITIALIZATION
// =========================================================================================
function init3dTerrainEngine() {
  const container = document.getElementById('canvas-3d-terrain-container');
  if (!container || !window.THREE) return;

  // Scene
  scene3d = new THREE.Scene();
  scene3d.background = new THREE.Color(0x06080f);
  scene3d.fog = new THREE.FogExp2(0x06080f, 0.012);

  // Camera
  const aspect = container.clientWidth / (container.clientHeight || 500);
  camera3d = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
  const cfg = TERRAIN_PRESETS[activePresetKey] || TERRAIN_PRESETS.nh10;
  camera3d.position.set(cfg.cameraPos[0], cfg.cameraPos[1], cfg.cameraPos[2]);

  // Renderer
  renderer3d = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance", preserveDrawingBuffer: true });
  renderer3d.setSize(container.clientWidth, container.clientHeight || 500);
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
    controls3d.maxDistance = 150;
    controls3d.target.set(0, 8, 0);
  }

  // Lighting
  const ambientLight = new THREE.AmbientLight(0x8cb0d8, 0.70);
  scene3d.add(ambientLight);

  const sunLight = new THREE.DirectionalLight(0xfff3d6, 1.3);
  sunLight.position.set(50, 85, 45);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.width = 1024;
  sunLight.shadow.mapSize.height = 1024;
  scene3d.add(sunLight);

  const fillLight = new THREE.DirectionalLight(0x427bbb, 0.50);
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

  // Re-inject HUD elements inside canvas container if needed
  injectCanvasHudElements(container);

  // Resize handler
  window.addEventListener('resize', on3dWindowResize);

  terrain3dInitialized = true;
  animate3dLoop();
}

function injectCanvasHudElements(container) {
  if (container.querySelector('#hud-3d-live-badge')) return;

  const hudHtml = `
    <!-- Overlay Navigation Helper Badge -->
    <div class="absolute top-3 left-3 z-10 pointer-events-none p-2.5 bg-black/80 backdrop-blur-md rounded-2xl border border-zinc-800/80 text-[10px] font-mono text-zinc-300 space-y-0.5 shadow-xl">
      <div class="text-cyan-400 font-bold flex items-center space-x-1.5">
        <span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
        <span>3D Orbit Controls:</span>
      </div>
      <div>Left Drag: Rotate 360° | Right Drag: Pan</div>
      <div>Scroll / Pinch: Zoom Free Camera</div>
    </div>

    <!-- Live Real-Time Stream HUD Badge (WeatherAndRadar.in) -->
    <div id="hud-3d-live-badge" class="hidden absolute top-16 left-3 z-10 p-2 sm:p-2.5 bg-emerald-950/90 border border-emerald-500/60 rounded-xl text-[11px] font-mono text-emerald-300 backdrop-blur-md shadow-xl flex items-center space-x-2">
      <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
      <span id="hud-3d-live-text">Streaming Live WeatherAndRadar.in & Ambee Telemetry</span>
    </div>

    <!-- Live AI Multi-Hazard Risk HUD: Risk = Hazard * Exposure -->
    <div class="absolute bottom-3 left-3 z-10 p-3 bg-black/85 backdrop-blur-md rounded-2xl border border-zinc-800 text-xs font-mono text-zinc-300 space-y-1.5 max-w-[280px] sm:max-w-xs shadow-2xl">
      <div class="flex items-center justify-between border-b border-zinc-800 pb-1">
        <span class="text-cyan-400 font-bold flex items-center space-x-1.5">
          <span class="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
          <span>Risk Engine: R = H × DEI</span>
        </span>
        <span class="text-[9px] px-1.5 py-0.5 rounded bg-zinc-900 text-zinc-300 border border-zinc-700 font-bold">XGBoost+LSTM</span>
      </div>
      <div class="grid grid-cols-2 gap-1.5 text-[11px]">
        <div>Hazard (H): <b class="text-amber-400 font-bold" id="hud-3d-hazard">0.892</b></div>
        <div>Exposure (DEI): <b class="text-cyan-300 font-bold" id="hud-3d-dei">0.684</b></div>
        <div class="col-span-2 pt-1 border-t border-zinc-800 flex items-center justify-between">
          <span>Calculated Risk (R):</span>
          <b class="text-rose-400 text-sm font-black font-mono" id="hud-3d-risk-score">0.610</b>
        </div>
      </div>
      <div class="grid grid-cols-2 gap-1 text-[10px] text-zinc-400 pt-1 border-t border-zinc-800">
        <div>Est. Volume: <b class="text-white" id="hud-3d-volume">16,400 m³</b></div>
        <div>Runout Speed: <b class="text-red-400" id="hud-3d-speed">7.8 m/s</b></div>
      </div>
    </div>

    <!-- 3D Legend Badge -->
    <div class="absolute top-3 right-3 z-10 pointer-events-none p-2.5 bg-black/80 backdrop-blur-md rounded-2xl border border-zinc-800/80 text-[10px] font-mono space-y-1 shadow-xl">
      <div class="flex items-center space-x-1.5">
        <span class="w-2.5 h-2.5 rounded-full bg-red-500"></span>
        <span class="text-zinc-200">Landslide Slip Wedge</span>
      </div>
      <div class="flex items-center space-x-1.5">
        <span class="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
        <span class="text-zinc-200">Piezometer (Pore Water)</span>
      </div>
      <div class="flex items-center space-x-1.5">
        <span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
        <span class="text-zinc-200">Borehole Inclinometer</span>
      </div>
    </div>
  `;

  const wrapper = document.createElement('div');
  wrapper.innerHTML = hudHtml;
  while (wrapper.firstChild) {
    container.appendChild(wrapper.firstChild);
  }
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

function buildFlashFloodDebrisSystem() {
  const floodCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(12, 26, -22),
    new THREE.Vector3(10, 18, -10),
    new THREE.Vector3(6, 9, 2),
    new THREE.Vector3(4, 4.5, 9),
    new THREE.Vector3(2, 2.0, 22),
    new THREE.Vector3(0, 1.3, 35)
  ]);

  const floodGeom = new THREE.TubeGeometry(floodCurve, 50, 2.2, 8, false);
  const floodMat = new THREE.MeshStandardMaterial({
    color: 0x06b6d4,
    roughness: 0.15,
    metalness: 0.85,
    transparent: true,
    opacity: 0.8
  });
  floodTorrentMesh = new THREE.Mesh(floodGeom, floodMat);
  floodTorrentMesh.visible = true;
  scene3d.add(floodTorrentMesh);

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

function buildSoilErosionRills() {
  const rillMaterial = new THREE.LineBasicMaterial({ color: 0x78350f, linewidth: 2 });
  const rillPaths = [
    [new THREE.Vector3(-14, 22, -18), new THREE.Vector3(-12, 14, -6), new THREE.Vector3(-11, 7, 6)],
    [new THREE.Vector3(-18, 20, -15), new THREE.Vector3(-16, 11, -3), new THREE.Vector3(-15, 6, 7)],
    [new THREE.Vector3(18, 24, -20), new THREE.Vector3(16, 16, -8), new THREE.Vector3(14, 8, 4)]
  ];

  rillPaths.forEach(pts => {
    const geom = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geom, rillMaterial);
    soilErosionRills.push(line);
    scene3d.add(line);
  });
}

function buildGeotechSensorPins() {
  const pinData = [
    { x: -5, y: 15, z: -5, color: 0x06b6d4, type: 'Piezometer P-1' },
    { x: 3, y: 13, z: -3, color: 0x06b6d4, type: 'Piezometer P-2' },
    { x: -2, y: 8, z: 4, color: 0x10b981, type: 'Inclinometer INC-01' },
    { x: 12, y: 16, z: -10, color: 0x10b981, type: 'Inclinometer INC-02' }
  ];

  pinData.forEach(d => {
    const sphereGeom = new THREE.SphereGeometry(0.65, 16, 16);
    const sphereMat = new THREE.MeshStandardMaterial({
      color: d.color,
      emissive: d.color,
      emissiveIntensity: 0.4,
      metalness: 0.8,
      roughness: 0.2
    });
    const sphere = new THREE.Mesh(sphereGeom, sphereMat);
    sphere.position.set(d.x, d.y, d.z);
    scene3d.add(sphere);

    const stemGeom = new THREE.CylinderGeometry(0.08, 0.08, 3.5, 8);
    const stemMat = new THREE.MeshBasicMaterial({ color: 0x64748b });
    const stem = new THREE.Mesh(stemGeom, stemMat);
    stem.position.set(d.x, d.y - 1.75, d.z);
    scene3d.add(stem);

    sensorPins.push({ sphere, baseY: d.y, type: d.type });
  });
}

function buildRainParticleSystem() {
  const rainCount = 1200;
  const geom = new THREE.BufferGeometry();
  const positions = new Float32Array(rainCount * 3);

  for (let i = 0; i < rainCount * 3; i += 3) {
    positions[i] = (Math.random() - 0.5) * 110;
    positions[i + 1] = Math.random() * 55;
    positions[i + 2] = (Math.random() - 0.5) * 110;
  }
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const mat = new THREE.PointsMaterial({
    color: 0x93c5fd,
    size: 0.35,
    transparent: true,
    opacity: 0.65
  });

  rainParticles = new THREE.Points(geom, mat);
  scene3d.add(rainParticles);
}

// =========================================================================================
// 3D ANIMATION LOOP & DRONE FLYOVER
// =========================================================================================
function animate3dLoop() {
  animFrameId = requestAnimationFrame(animate3dLoop);

  // Cinematic Drone Orbit Animation
  if (isDroneOrbiting && camera3d) {
    droneOrbitAngle += 0.005;
    const orbitRadius = 65;
    camera3d.position.x = Math.cos(droneOrbitAngle) * orbitRadius;
    camera3d.position.z = Math.sin(droneOrbitAngle) * orbitRadius;
    camera3d.position.y = 35 + Math.sin(droneOrbitAngle * 2) * 6;
    camera3d.lookAt(0, 8, 0);
    if (controls3d) controls3d.update();
  } else if (controls3d) {
    controls3d.update();
  }

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

  // Flash flood debris particles rushing downhill
  if (floodParticles && (currentHazardMode === 'all' || currentHazardMode === 'flood')) {
    const fpos = floodParticles.geometry.attributes.position.array;
    const fSpeed = (current3dRain / 100) * 0.8 + 0.3;
    for (let i = 2; i < fpos.length; i += 3) {
      fpos[i] += fSpeed;
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
      landslideWedge.material.color.setHex(0xef4444);
      landslideWedge.material.opacity = 0.9;
    } else if (instability > 0.1) {
      landslideWedge.material.color.setHex(0xf59e0b);
      landslideWedge.material.opacity = 0.8;
    } else {
      landslideWedge.material.color.setHex(0x10b981);
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

  const width = container.clientWidth;
  const height = container.clientHeight || 500;
  camera3d.aspect = width / height;
  camera3d.updateProjectionMatrix();
  renderer3d.setSize(width, height);
}

// =========================================================================================
// MULTI-HAZARD MODE & SIMULATION CONTROLS
// =========================================================================================
function set3dHazardMode(mode) {
  currentHazardMode = mode;
  ['all', 'landslide', 'flood', 'erosion'].forEach(m => {
    const btn = document.getElementById(`btn-3d-mode-${m}`);
    if (btn) {
      if (m === mode) {
        btn.className = 'px-2.5 py-1 rounded-lg bg-cyan-600 text-white font-bold text-xs transition shadow';
      } else {
        btn.className = 'px-2.5 py-1 rounded-lg text-zinc-400 hover:text-white text-xs transition';
      }
    }
  });

  update3dSimulationFromSlider(current3dRain);
}

function update3dSimulationFromSlider(rainVal) {
  current3dRain = parseFloat(rainVal);
  current3dPore = Math.max(8.0, (current3dRain * 0.38) + 5.0);

  const cfg = TERRAIN_PRESETS[activePresetKey] || TERRAIN_PRESETS.nh10;

  const p_xgb = 1.0 / (1.0 + Math.exp(-0.024 * (current3dRain - 130.0)));
  const p_lstm = 1.0 / (1.0 + Math.exp(-0.06 * (current3dPore - 28.0)));
  const hazard = Math.min(0.99, (0.55 * p_xgb) + (0.35 * p_lstm) + (0.10 * p_xgb * p_lstm));

  const dei = Math.min(0.95, (cfg.popDensity / 1000.0) * (1.0 + cfg.vulnerabilityRatio) * 0.85);
  const risk = Math.min(0.99, hazard * dei);
  const fsVal = Math.max(0.68, 1.85 - (current3dRain * 0.0058)).toFixed(2);

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

  const estVol = Math.round(hazard * 18500);
  const estSpeed = (hazard * 9.2).toFixed(1);
  if (hudVol) hudVol.innerText = `${estVol.toLocaleString()} m³`;
  if (hudSpeed) hudSpeed.innerText = `${estSpeed} m/s`;

  if (fsLabel) {
    fsLabel.innerText = fsVal;
    if (parseFloat(fsVal) < 1.0 || risk >= 0.55) {
      fsLabel.className = 'text-xl font-black font-mono text-rose-500 animate-pulse';
      if (statusLabel) {
        statusLabel.innerHTML = `<span class="px-2 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800 font-black">COLLAPSE / DETACHMENT ACTIVE (Risk: ${(risk*100).toFixed(0)}%)</span>`;
      }
    } else if (parseFloat(fsVal) < 1.2 || risk >= 0.30) {
      fsLabel.className = 'text-xl font-black font-mono text-amber-400';
      if (statusLabel) {
        statusLabel.innerHTML = `<span class="px-2 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800 font-black">ACCELERATING CREEP (Risk: ${(risk*100).toFixed(0)}%)</span>`;
      }
    } else {
      fsLabel.className = 'text-xl font-black font-mono text-emerald-400';
      if (statusLabel) {
        statusLabel.innerHTML = `<span class="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-black">SECURE & STABLE (Risk: ${(risk*100).toFixed(0)}%)</span>`;
      }
    }
  }
}

// =========================================================================================
// MODAL OPEN / CLOSE & PRESET SELECTION
// =========================================================================================
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
  if (isDroneOrbiting) toggleDroneOrbit();
}

function select3dPreset(presetKey) {
  activePresetKey = presetKey;
  const cfg = TERRAIN_PRESETS[presetKey] || TERRAIN_PRESETS.nh10;

  const titleEl = document.getElementById('modal-3d-title');
  const subEl = document.getElementById('modal-3d-subtitle');
  if (titleEl) titleEl.innerText = cfg.title;
  if (subEl) subEl.innerText = cfg.sub;

  const selector = document.getElementById('select-3d-preset');
  if (selector) selector.value = presetKey;

  if (camera3d && controls3d) {
    camera3d.position.set(cfg.cameraPos[0], cfg.cameraPos[1], cfg.cameraPos[2]);
    controls3d.target.set(0, 8, 0);
    controls3d.update();
  }

  const slider = document.getElementById('input-3d-rain');
  if (slider) {
    slider.value = '55';
    update3dSimulationFromSlider(55);
  }

  if (currentViewMode3D === 'google_earth') {
    renderGoogleEarth3DView();
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

// =========================================================================================
// CINEMATIC DRONE FLYOVER & VIEW CONTROLS
// =========================================================================================
function toggleDroneOrbit() {
  isDroneOrbiting = !isDroneOrbiting;
  const btn = document.getElementById('btn-3d-drone-orbit');
  if (btn) {
    if (isDroneOrbiting) {
      btn.className = 'px-2.5 py-1 rounded-xl bg-cyan-600 text-white font-bold text-xs transition shadow-lg shadow-cyan-600/30 flex items-center space-x-1 border border-cyan-400';
      btn.innerHTML = `<span class="w-2 h-2 rounded-full bg-white animate-ping"></span><span>Drone Flyover: ON</span>`;
    } else {
      btn.className = 'px-2.5 py-1 rounded-xl bg-zinc-800 text-zinc-300 hover:text-white text-xs font-semibold transition border border-zinc-700 flex items-center space-x-1';
      btn.innerHTML = `<i data-lucide="crosshair" class="w-3.5 h-3.5 text-cyan-400"></i><span>Drone Flyover</span>`;
      if (window.lucide) window.lucide.createIcons();
    }
  }
}

function capture3dSnapshot() {
  if (!renderer3d) return;
  try {
    const dataUrl = renderer3d.domElement.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = `MDoNER_3D_Twin_${activePresetKey}_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } catch (e) {
    console.warn('Snapshot capture warning:', e);
  }
}

function toggle3dFullscreen() {
  const modalBox = document.querySelector('#modal-3d-terrain > div');
  if (!modalBox) return;

  if (!document.fullscreenElement) {
    modalBox.requestFullscreen().catch(err => {
      console.warn('Fullscreen request error:', err);
    });
  } else {
    document.exitFullscreen();
  }
}

// =========================================================================================
// VIEW MODE SWITCHER: THREE.JS WEBGL VS. GOOGLE EARTH 3D TOPOGRAPHIC VIEW
// =========================================================================================
function switch3dViewMode(mode) {
  currentViewMode3D = mode;
  const webglCont = document.getElementById('canvas-3d-terrain-container');
  const gearthCont = document.getElementById('view-3d-google-earth');
  const footerSim = document.getElementById('footer-3d-sim-controls');
  const btnWebgl = document.getElementById('btn-tab-3d-webgl');
  const btnGearth = document.getElementById('btn-tab-3d-gearth');

  if (mode === 'google_earth') {
    if (webglCont) webglCont.classList.add('hidden');
    if (footerSim) footerSim.classList.add('hidden');
    if (gearthCont) gearthCont.classList.remove('hidden');

    if (btnWebgl) {
      btnWebgl.className = 'px-3 py-1.5 rounded-xl text-zinc-400 hover:text-white text-xs font-bold transition flex items-center space-x-1.5';
    }
    if (btnGearth) {
      btnGearth.className = 'px-3 py-1.5 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 text-white text-xs font-black transition shadow-lg shadow-blue-500/25 flex items-center space-x-1.5 border border-blue-400/50';
    }

    renderGoogleEarth3DView();
  } else {
    if (webglCont) webglCont.classList.remove('hidden');
    if (footerSim) footerSim.classList.remove('hidden');
    if (gearthCont) gearthCont.classList.add('hidden');

    if (btnWebgl) {
      btnWebgl.className = 'px-3 py-1.5 rounded-xl bg-cyan-600 text-white text-xs font-black transition shadow-lg shadow-cyan-500/25 flex items-center space-x-1.5 border border-cyan-400/50';
    }
    if (btnGearth) {
      btnGearth.className = 'px-3 py-1.5 rounded-xl text-zinc-400 hover:text-white text-xs font-bold transition flex items-center space-x-1.5';
    }

    on3dWindowResize();
  }
}

// =========================================================================================
// GOOGLE EARTH 3D TOPOGRAPHIC VIEW RENDERER & ACTIONS
// =========================================================================================
function renderGoogleEarth3DView() {
  const cfg = TERRAIN_PRESETS[activePresetKey] || TERRAIN_PRESETS.nh10;

  const corridorTitle = document.getElementById('gearth-corridor-title');
  const coordsLabel = document.getElementById('gearth-coords');
  const bedrockLabel = document.getElementById('gearth-bedrock');
  const foliationLabel = document.getElementById('gearth-foliation');
  const insarLabel = document.getElementById('gearth-insar');
  const fsLabel = document.getElementById('gearth-fs');
  const lifelineLabel = document.getElementById('gearth-lifeline');
  const mitigationLabel = document.getElementById('gearth-mitigation');

  if (corridorTitle) corridorTitle.innerText = `${cfg.state} • ${cfg.title}`;
  if (coordsLabel) coordsLabel.innerText = `${cfg.lat.toFixed(4)}° N, ${cfg.lon.toFixed(4)}° E • Alt: ${cfg.alt_m}m ASL`;
  if (bedrockLabel) bedrockLabel.innerText = cfg.bedrock;
  if (foliationLabel) foliationLabel.innerText = cfg.foliation;
  if (insarLabel) insarLabel.innerText = cfg.insar_rate;
  if (fsLabel) fsLabel.innerText = `FS = ${cfg.baseFS} (${cfg.baseFS < 1.0 ? 'Critical' : 'Moderate'})`;
  if (lifelineLabel) lifelineLabel.innerText = cfg.lifeline;
  if (mitigationLabel) mitigationLabel.innerText = cfg.recommendedMitigation;

  setTimeout(() => {
    initOrUpdateGearthMap(cfg);
  }, 100);
}

function initOrUpdateGearthMap(cfg) {
  const mapContainer = document.getElementById('gearth-satellite-map');
  if (!mapContainer || !window.L) return;

  if (!gearthSatelliteMap) {
    gearthSatelliteMap = L.map('gearth-satellite-map', {
      center: [cfg.lat, cfg.lon],
      zoom: 14,
      zoomControl: true
    });

    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Esri, Maxar, Earthstar Geographics',
      maxZoom: 18
    }).addTo(gearthSatelliteMap);

    L.tileLayer('https://tiles.wmflabs.org/hillshading/{z}/{x}/{y}.png', {
      maxZoom: 15,
      opacity: 0.35
    }).addTo(gearthSatelliteMap);

    gearthMarkerGroup = L.layerGroup().addTo(gearthSatelliteMap);
  } else {
    gearthSatelliteMap.setView([cfg.lat, cfg.lon], 14);
    gearthMarkerGroup.clearLayers();
  }

  const p1 = [cfg.lat - 0.003, cfg.lon - 0.003];
  const p2 = [cfg.lat - 0.003, cfg.lon + 0.003];
  const p3 = [cfg.lat + 0.003, cfg.lon + 0.004];
  const p4 = [cfg.lat + 0.003, cfg.lon - 0.004];

  const poly = L.polygon([p1, p2, p3, p4], {
    color: '#ef4444',
    weight: 2,
    fillColor: '#ef4444',
    fillOpacity: 0.25,
    dashArray: '4, 4'
  }).addTo(gearthMarkerGroup);

  poly.bindPopup(`
    <div style="font-family: sans-serif; font-size: 11px;">
      <b style="color: #ef4444;">${cfg.title}</b><br/>
      <b>Bedrock:</b> ${cfg.bedrock}<br/>
      <b>InSAR Creep:</b> ${cfg.insar_rate}<br/>
      <b>Factor of Safety:</b> ${cfg.baseFS}
    </div>
  `);

  const centerIcon = L.divIcon({
    className: 'gearth-center-icon',
    html: `<div style="width:16px;height:16px;border-radius:50%;background:#06b6d4;border:3px solid white;box-shadow:0 0 10px #06b6d4;"></div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8]
  });

  L.marker([cfg.lat, cfg.lon], { icon: centerIcon })
    .addTo(gearthMarkerGroup)
    .bindPopup(`<b>3D Camera Focus:</b><br/>${cfg.lat}° N, ${cfg.lon}° E<br/>Elevation: ${cfg.alt_m} m`);

  gearthSatelliteMap.invalidateSize();
}

function openGoogleEarthWeb() {
  const cfg = TERRAIN_PRESETS[activePresetKey] || TERRAIN_PRESETS.nh10;
  window.open(cfg.gearth_url, '_blank', 'noopener,noreferrer');
}

function downloadGeotechnicalKML() {
  const cfg = TERRAIN_PRESETS[activePresetKey] || TERRAIN_PRESETS.nh10;
  const kmlContent = `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
  <Document>
    <name>MDoNER 3D Landslide Hazard Corridor - ${cfg.title}</name>
    <description>Ministry of Development of North Eastern Region (MDoNER) 3D Geotechnical Hazard Zone</description>
    <LookAt>
      <longitude>${cfg.lon}</longitude>
      <latitude>${cfg.lat}</latitude>
      <altitude>${cfg.alt_m}</altitude>
      <heading>${cfg.heading}</heading>
      <tilt>${cfg.tilt}</tilt>
      <range>${cfg.range}</range>
      <altitudeMode>relativeToGround</altitudeMode>
    </LookAt>
    <Placemark>
      <name>${cfg.title} - Headscarp</name>
      <description><![CDATA[
        <b>State:</b> ${cfg.state}<br/>
        <b>Bedrock Formation:</b> ${cfg.bedrock}<br/>
        <b>Foliation:</b> ${cfg.foliation}<br/>
        <b>Spaceborne InSAR LOS Velocity:</b> ${cfg.insar_rate}<br/>
        <b>Factor of Safety (FS):</b> ${cfg.baseFS}<br/>
        <b>Lifeline:</b> ${cfg.lifeline}<br/>
        <b>Recommended Mitigation:</b> ${cfg.recommendedMitigation}
      ]]></description>
      <Point>
        <coordinates>${cfg.lon},${cfg.lat},${cfg.alt_m}</coordinates>
      </Point>
    </Placemark>
    <Placemark>
      <name>Critical Slip Surface Wedge</name>
      <Style>
        <LineStyle><color>ff0000ff</color><width>3</width></LineStyle>
        <PolyStyle><color>7f0000ff</color></PolyStyle>
      </Style>
      <Polygon>
        <extrude>1</extrude>
        <altitudeMode>relativeToGround</altitudeMode>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              ${cfg.lon - 0.003},${cfg.lat - 0.002},${cfg.alt_m + 30}
              ${cfg.lon + 0.003},${cfg.lat - 0.002},${cfg.alt_m + 30}
              ${cfg.lon + 0.004},${cfg.lat + 0.003},${cfg.alt_m - 20}
              ${cfg.lon - 0.004},${cfg.lat + 0.003},${cfg.alt_m - 20}
              ${cfg.lon - 0.003},${cfg.lat - 0.002},${cfg.alt_m + 30}
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>`;

  const blob = new Blob([kmlContent], { type: 'application/vnd.google-earth.kml+xml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `MDoNER_Geotech_3D_${cfg.key.toUpperCase()}.kml`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function copyGeotechCoords() {
  const cfg = TERRAIN_PRESETS[activePresetKey] || TERRAIN_PRESETS.nh10;
  const text = `${cfg.lat.toFixed(5)}, ${cfg.lon.toFixed(5)} (Alt: ${cfg.alt_m}m)`;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('btn-copy-coords');
    if (btn) {
      const orig = btn.innerHTML;
      btn.innerHTML = `<i data-lucide="check" class="w-3.5 h-3.5 text-emerald-400"></i><span class="text-emerald-400">Copied!</span>`;
      if (window.lucide) window.lucide.createIcons();
      setTimeout(() => {
        btn.innerHTML = orig;
        if (window.lucide) window.lucide.createIcons();
      }, 2000);
    }
  });
}

// =========================================================================================
// WEATHERANDRADAR.IN LIVE RAINFALL STREAMING ENGINE
// =========================================================================================
let is3dLiveRainStreaming = false;
let live3dStreamInterval = null;
let lastLiveRainData = null;

async function toggle3dLiveRainStreaming() {
  is3dLiveRainStreaming = !is3dLiveRainStreaming;
  const btn = document.getElementById('btn-3d-live-stream');
  const badge = document.getElementById('hud-3d-live-badge');

  if (is3dLiveRainStreaming) {
    if (btn) {
      btn.className = "px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-black text-xs transition shadow-lg shadow-emerald-600/40 flex items-center space-x-1.5 border border-emerald-400";
      btn.innerHTML = `
        <span class="w-2 h-2 rounded-full bg-white animate-ping"></span>
        <span class="hidden sm:inline">LIVE STREAM: ON</span>
        <span class="sm:hidden">LIVE: ON</span>
      `;
    }
    if (badge) badge.classList.remove('hidden');

    await fetchAndApplyLiveRainfall3D();
    if (!live3dStreamInterval) {
      live3dStreamInterval = setInterval(fetchAndApplyLiveRainfall3D, 30000);
    }
  } else {
    if (btn) {
      btn.className = "px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-300 font-bold text-xs transition border border-zinc-700 flex items-center space-x-1.5";
      btn.innerHTML = `
        <span class="w-2 h-2 rounded-full bg-zinc-500"></span>
        <span class="hidden sm:inline">LIVE STREAM: OFF</span>
        <span class="sm:hidden">LIVE</span>
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
    haflong: 'assam',
    sela: 'arunachal',
    noney: 'manipur',
    hunthar: 'mizoram',
    paglapahar: 'nagaland',
    baramura: 'tripura'
  };
  const reg = regMap[activePresetKey] || (window.currentRegion || 'sikkim');

  try {
    const res = await fetch(`http://localhost:8000/weather/live-rainfall?region=${reg}`);
    if (!res.ok) throw new Error("Backend live-rainfall unavailable");
    const data = await res.json();
    lastLiveRainData = data;

    const liveRate = data.live_rainfall_rate_mm_h || 1.2;
    const prob = (data.precipitation_probability || 30) / 100.0;
    const effectiveSimRain = Math.round(Math.min(220, Math.max(35, (liveRate * 18.0) + (prob * 65.0) + 25.0)));

    const slider = document.getElementById('input-3d-rain');
    if (slider) {
      slider.value = effectiveSimRain.toString();
    }
    update3dSimulationFromSlider(effectiveSimRain);

    const liveStatusText = document.getElementById('hud-3d-live-text');
    if (liveStatusText) {
      liveStatusText.innerHTML = `
        <span class="text-cyan-400 font-bold">${data.city} (${data.temperature_c}°C)</span>:
        Live Rain <b class="text-white">${liveRate} mm/h</b> • Humidity <b class="text-white">${data.humidity_pct}%</b> • Prob <b class="text-white">${data.precipitation_probability}%</b>
        <span class="text-[9px] text-zinc-400 ml-1 block sm:inline">[WeatherAndRadar.in & Ambee]</span>
      `;
    }

    const liveBadge = document.getElementById('hud-3d-live-badge');
    if (liveBadge) liveBadge.classList.remove('hidden');

  } catch (e) {
    console.warn("Live 3D rainfall fetch error:", e);
  }
}
