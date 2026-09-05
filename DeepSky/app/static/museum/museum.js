import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";
import { addCosmicArchitecture } from "/static/museum/cosmic.js";

const canvas = document.querySelector("#museum-canvas");
const intro = document.querySelector("#intro");
const enterButton = document.querySelector("#enter-museum");
const exitButton = document.querySelector("#exit-gallery");
const hud = document.querySelector("#hud");
const mobileControls = document.querySelector("#mobile-controls");
const loading = document.querySelector("#loading");
const loadingProgress = document.querySelector("#loading-progress");
const dialog = document.querySelector("#exhibit-dialog");
const dialogClose = document.querySelector("#dialog-close");

const exhibits = [
  {
    title: "Andromeda Galaxy",
    category: "GALAXY · M31",
    description: "Our nearest large galactic neighbor, presented with its bright nucleus, dark dust lanes, and satellite galaxies preserved from the captured signal.",
    src: "/static/museum/images/andromeda-m31.png",
    position: [0, 2.75, -10.35],
    rotation: [0, 0, 0],
    height: 5.5,
  },
  {
    title: "Bode's Galaxy",
    category: "GALAXY · M81",
    description: "A low-signal observation of Messier 81, carefully balanced to retain the galaxy's soft outer structure and natural stellar field.",
    src: "/static/museum/images/m81.png",
    position: [-8.15, 2.65, -2.1],
    rotation: [0, Math.PI / 2, 0],
    height: 5.1,
  },
  {
    title: "Veil Nebula",
    category: "NEBULA · NGC 6992",
    description: "Delicate oxygen and hydrogen filaments from a supernova remnant, rendered in cyan and warm orange against the surrounding star field.",
    src: "/static/museum/images/veil-nebula.png",
    position: [8.15, 2.65, -2.1],
    rotation: [0, -Math.PI / 2, 0],
    height: 5.1,
  },
];

let renderer;
let scene;
let camera;
let interactive = false;
let yaw = 0;
let pitch = -0.04;
let dragging = false;
let moved = false;
let pointerStart = { x: 0, y: 0 };
let lastPointer = { x: 0, y: 0 };
let hoveredFrame = null;
let updateCosmos = null;
const keys = new Set();
const moveButtons = new Set();
const clickableFrames = [];
const clock = new THREE.Clock();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function createLabelTexture(title, category) {
  const labelCanvas = document.createElement("canvas");
  labelCanvas.width = 1024;
  labelCanvas.height = 210;
  const context = labelCanvas.getContext("2d");
  context.clearRect(0, 0, 1024, 210);
  context.fillStyle = "rgba(5, 7, 11, .94)";
  context.fillRect(0, 0, 1024, 210);
  context.fillStyle = "#55b9ff";
  context.font = "600 30px Arial";
  context.letterSpacing = "8px";
  context.fillText(category, 48, 65);
  context.fillStyle = "#f5f7fa";
  context.font = "300 58px Arial";
  context.fillText(title.toUpperCase(), 48, 145);
  const texture = new THREE.CanvasTexture(labelCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function addLightStrip(position, scale, rotation = [0, 0, 0]) {
  const strip = new THREE.Mesh(
    new THREE.BoxGeometry(...scale),
    new THREE.MeshBasicMaterial({ color: 0xd49b5f, toneMapped: false })
  );
  strip.position.set(...position);
  strip.rotation.set(...rotation);
  scene.add(strip);
}

function addExhibit(texture, exhibit) {
  const ratio = texture.image.width / texture.image.height;
  const width = exhibit.height * ratio;
  const group = new THREE.Group();
  group.position.set(...exhibit.position);
  group.rotation.set(...exhibit.rotation);

  const back = new THREE.Mesh(
    new THREE.BoxGeometry(width + .42, exhibit.height + .42, .24),
    new THREE.MeshStandardMaterial({ color: 0x080a0e, metalness: .72, roughness: .28 })
  );
  back.position.z = -.13;
  group.add(back);

  const art = new THREE.Mesh(
    new THREE.PlaneGeometry(width, exhibit.height),
    new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide })
  );
  art.userData.exhibit = exhibit;
  art.position.z = .01;
  group.add(art);
  clickableFrames.push(art);

  const labelRatio = 1024 / 210;
  const labelWidth = Math.min(width, 3.8);
  const label = new THREE.Mesh(
    new THREE.PlaneGeometry(labelWidth, labelWidth / labelRatio),
    new THREE.MeshBasicMaterial({ map: createLabelTexture(exhibit.title, exhibit.category), transparent: true })
  );
  label.position.set(0, -(exhibit.height / 2) - .62, .025);
  group.add(label);

  const glow = new THREE.PointLight(0xdba563, 8, 5.5, 2);
  glow.position.set(0, -(exhibit.height / 2) - .15, 1.25);
  group.add(glow);
  scene.add(group);
}

function buildRoom() {
  const roomMaterial = new THREE.MeshStandardMaterial({ color: 0x090c12, roughness: .48, metalness: .48 });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(22, 25), new THREE.MeshPhysicalMaterial({ color: 0x080b10, roughness: .22, metalness: .72, clearcoat: .45 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.z = -.5;
  scene.add(floor);

  const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(22, 25), roomMaterial);
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.set(0, 6.25, -.5);
  scene.add(ceiling);

  const backWall = new THREE.Mesh(new THREE.BoxGeometry(18, 6.25, .4), roomMaterial);
  backWall.position.set(0, 3.1, -10.65);
  scene.add(backWall);
  const leftWall = new THREE.Mesh(new THREE.BoxGeometry(.4, 6.25, 20.5), roomMaterial);
  leftWall.position.set(-8.45, 3.1, -.55);
  scene.add(leftWall);
  const rightWall = leftWall.clone();
  rightWall.position.x = 8.45;
  scene.add(rightWall);

  for (const x of [-6.8, -3.4, 0, 3.4, 6.8]) {
    addLightStrip([x, .04, -10.35], [2.5, .035, .06]);
  }
  for (const z of [-8.5, -5, -1.5, 2]) {
    addLightStrip([-8.17, .04, z], [.06, .035, 2.35]);
    addLightStrip([8.17, .04, z], [.06, .035, 2.35]);
  }

  const plinth = new THREE.Mesh(
    new THREE.CylinderGeometry(2.15, 2.55, .45, 64),
    new THREE.MeshStandardMaterial({ color: 0x11151d, roughness: .25, metalness: .8 })
  );
  plinth.position.set(0, .23, -3.4);
  scene.add(plinth);
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(2.3, .035, 8, 96),
    new THREE.MeshBasicMaterial({ color: 0xdca560 })
  );
  ring.rotation.x = Math.PI / 2;
  ring.position.set(0, .48, -3.4);
  scene.add(ring);

  const sculpture = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1.15, 2),
    new THREE.MeshStandardMaterial({ color: 0x151b24, roughness: .72, metalness: .2, flatShading: true })
  );
  sculpture.scale.y = .38;
  sculpture.position.set(0, .78, -3.4);
  scene.add(sculpture);

  scene.add(new THREE.HemisphereLight(0x6f91b5, 0x17100b, 1.2));
  const ceilingLight = new THREE.RectAreaLight(0xa9d8ff, 4, 6, 2);
  ceilingLight.position.set(0, 5.8, -4.5);
  ceilingLight.rotation.x = -Math.PI / 2;
  scene.add(ceilingLight);
}

function addStars() {
  const positions = [];
  for (let i = 0; i < 900; i += 1) {
    const radius = 35 + Math.random() * 25;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(THREE.MathUtils.randFloatSpread(2));
    positions.push(radius * Math.sin(phi) * Math.cos(theta), radius * Math.cos(phi), radius * Math.sin(phi) * Math.sin(theta));
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0xb9dbff, size: .06, transparent: true, opacity: .75 })));
}

function openExhibit(exhibit) {
  document.querySelector("#dialog-image").src = exhibit.src;
  document.querySelector("#dialog-image").alt = `${exhibit.title} full-resolution astrophotography exhibit`;
  document.querySelector("#dialog-category").textContent = exhibit.category;
  document.querySelector("#dialog-title").textContent = exhibit.title;
  document.querySelector("#dialog-description").textContent = exhibit.description;
  document.querySelector("#dialog-original").href = exhibit.src;
  dialog.showModal();
}

function updateRaycast(event) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(clickableFrames, false)[0]?.object || null;
  if (hoveredFrame !== hit) {
    if (hoveredFrame) hoveredFrame.material.color.setHex(0xffffff);
    hoveredFrame = hit;
    if (hoveredFrame) hoveredFrame.material.color.setHex(0xb9ddff);
    canvas.style.cursor = hit ? "pointer" : dragging ? "grabbing" : "grab";
  }
  return hit;
}

function setInteractive(enabled) {
  interactive = enabled;
  intro.classList.toggle("is-hidden", enabled);
  hud.classList.toggle("is-visible", enabled);
  mobileControls.classList.toggle("is-visible", enabled);
  exitButton.classList.toggle("is-visible", enabled);
  if (!enabled) {
    camera.position.set(0, 1.7, 7.6);
    yaw = 0;
    pitch = -0.04;
  }
}

function bindControls() {
  enterButton.addEventListener("click", () => setInteractive(true));
  exitButton.addEventListener("click", () => setInteractive(false));
  dialogClose.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
  window.addEventListener("keydown", (event) => {
    if (["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.code)) {
      keys.add(event.code);
      event.preventDefault();
    }
    if (event.code === "Escape" && interactive && !dialog.open) setInteractive(false);
  });
  window.addEventListener("keyup", (event) => keys.delete(event.code));

  canvas.addEventListener("pointerdown", (event) => {
    if (!interactive || dialog.open) return;
    dragging = true;
    moved = false;
    pointerStart = { x: event.clientX, y: event.clientY };
    lastPointer = { ...pointerStart };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!interactive || dialog.open) return;
    if (dragging) {
      const dx = event.clientX - lastPointer.x;
      const dy = event.clientY - lastPointer.y;
      if (Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 6) moved = true;
      yaw -= dx * .0032;
      pitch = THREE.MathUtils.clamp(pitch - dy * .0026, -.58, .48);
      lastPointer = { x: event.clientX, y: event.clientY };
    } else {
      updateRaycast(event);
    }
  });
  canvas.addEventListener("pointerup", (event) => {
    if (!interactive) return;
    dragging = false;
    if (!moved) {
      const hit = updateRaycast(event);
      if (hit?.userData.exhibit) openExhibit(hit.userData.exhibit);
    }
  });
  canvas.addEventListener("wheel", (event) => {
    if (!interactive || dialog.open) return;
    const direction = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw));
    camera.position.addScaledVector(direction, THREE.MathUtils.clamp(event.deltaY * .002, -.6, .6));
    event.preventDefault();
  }, { passive: false });

  document.querySelectorAll("[data-move]").forEach((button) => {
    const action = button.dataset.move;
    const start = (event) => { event.preventDefault(); moveButtons.add(action); };
    const end = () => moveButtons.delete(action);
    button.addEventListener("pointerdown", start);
    button.addEventListener("pointerup", end);
    button.addEventListener("pointercancel", end);
    button.addEventListener("pointerleave", end);
  });
}

function moveCamera(delta) {
  if (!interactive || dialog.open) return;
  const speed = 3.4 * delta;
  const forward = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw));
  const right = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
  if (keys.has("KeyW") || keys.has("ArrowUp") || moveButtons.has("forward")) camera.position.addScaledVector(forward, speed);
  if (keys.has("KeyS") || keys.has("ArrowDown") || moveButtons.has("back")) camera.position.addScaledVector(forward, -speed);
  if (keys.has("KeyA") || keys.has("ArrowLeft") || moveButtons.has("left")) camera.position.addScaledVector(right, -speed);
  if (keys.has("KeyD") || keys.has("ArrowRight") || moveButtons.has("right")) camera.position.addScaledVector(right, speed);
  camera.position.x = THREE.MathUtils.clamp(camera.position.x, -7.4, 7.4);
  camera.position.z = THREE.MathUtils.clamp(camera.position.z, -9.1, 8.2);
  camera.position.y = 1.7;
}

function resize() {
  const width = window.innerWidth;
  const height = document.querySelector("#museum-stage").clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.6));
  renderer.setSize(width, height, false);
}

function animate() {
  const delta = Math.min(clock.getDelta(), .05);
  moveCamera(delta);
  camera.rotation.order = "YXZ";
  camera.rotation.y = yaw;
  camera.rotation.x = pitch;
  if (updateCosmos) updateCosmos(clock.elapsedTime);
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

function init() {
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  } catch (error) {
    loading.innerHTML = "3D GALLERY UNAVAILABLE — VIEW THE COLLECTION BELOW";
    canvas.hidden = true;
    return;
  }
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x030508);
  scene.fog = new THREE.FogExp2(0x030508, .014);
  camera = new THREE.PerspectiveCamera(64, 1, .1, 120);
  camera.position.set(0, 1.7, 7.6);

  buildRoom();
  addStars();
  updateCosmos = addCosmicArchitecture(scene);
  bindControls();
  resize();
  window.addEventListener("resize", resize);

  const manager = new THREE.LoadingManager();
  manager.onProgress = (_url, loaded, total) => { loadingProgress.style.width = `${Math.round((loaded / total) * 100)}%`; };
  manager.onLoad = () => loading.classList.add("is-done");
  manager.onError = (url) => { loading.textContent = `AN EXHIBIT COULD NOT LOAD: ${url}`; };
  const loader = new THREE.TextureLoader(manager);
  exhibits.forEach((exhibit) => {
    loader.load(exhibit.src, (texture) => {
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
      addExhibit(texture, exhibit);
    });
  });
  animate();
}

init();
