import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";

function cosmicTexture(kind = "stars") {
  const size = 1024;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  context.fillStyle = "#010207";
  context.fillRect(0, 0, size, size);

  if (kind === "galaxy") {
    const core = context.createRadialGradient(512, 512, 0, 512, 512, 440);
    core.addColorStop(0, "rgba(255,247,222,.98)");
    core.addColorStop(.05, "rgba(255,199,145,.9)");
    core.addColorStop(.2, "rgba(102,137,255,.38)");
    core.addColorStop(.62, "rgba(68,82,190,.05)");
    core.addColorStop(1, "rgba(0,0,0,0)");
    context.fillStyle = core;
    context.fillRect(0, 0, size, size);
    context.globalCompositeOperation = "screen";
    for (let i = 0; i < 7000; i += 1) {
      const radius = Math.pow(Math.random(), .68) * 440;
      const angle = (i % 3) * (Math.PI * 2 / 3) + radius * .022 + THREE.MathUtils.randFloatSpread(.48);
      const x = 512 + Math.cos(angle) * radius;
      const y = 512 + Math.sin(angle) * radius * (.4 + Math.random() * .16);
      const brightness = 95 + Math.floor(Math.random() * 160);
      context.fillStyle = `rgba(${brightness},${Math.min(255, brightness + 18)},255,${.12 + Math.random() * .55})`;
      context.beginPath();
      context.arc(x, y, Math.random() < .025 ? 2.2 : .55 + Math.random() * 1.15, 0, Math.PI * 2);
      context.fill();
    }
  } else {
    const haze = context.createRadialGradient(510, 480, 30, 510, 480, 620);
    haze.addColorStop(0, "rgba(34,71,130,.22)");
    haze.addColorStop(.45, "rgba(45,22,92,.11)");
    haze.addColorStop(1, "rgba(0,0,0,0)");
    context.fillStyle = haze;
    context.fillRect(0, 0, size, size);
    for (let i = 0; i < 4200; i += 1) {
      const x = Math.random() * size;
      const y = Math.random() * size;
      const warm = Math.random() < .14;
      context.fillStyle = warm ? `rgba(255,210,175,${.35 + Math.random() * .6})` : `rgba(205,225,255,${.3 + Math.random() * .65})`;
      context.beginPath();
      context.arc(x, y, Math.random() < .035 ? 3.8 : .7 + Math.random() * 1.8, 0, Math.PI * 2);
      context.fill();
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export function addCosmicArchitecture(scene) {
  const group = new THREE.Group();
  const galaxy = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: cosmicTexture("galaxy"), transparent: true, opacity: .95, depthWrite: false, blending: THREE.AdditiveBlending, rotation: -.3 })
  );
  galaxy.position.set(0, 10.5, -10);
  galaxy.scale.set(17, 9, 1);
  galaxy.renderOrder = 2;
  group.add(galaxy);

  const glow = new THREE.PointLight(0x728dff, 17, 15, 2);
  glow.position.set(0, 7.5, -4.2);
  group.add(glow);

  const positions = [];
  const colors = [];
  const cool = new THREE.Color(0x9acbff);
  const warm = new THREE.Color(0xffd2a0);
  for (let i = 0; i < 3600; i += 1) {
    const radius = 18 + Math.random() * 25;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(THREE.MathUtils.randFloatSpread(2));
    positions.push(
      radius * Math.sin(phi) * Math.cos(theta),
      radius * Math.cos(phi) + 4,
      radius * Math.sin(phi) * Math.sin(theta)
    );
    const color = Math.random() < .12 ? warm : cool;
    colors.push(color.r, color.g, color.b);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  group.add(new THREE.Points(geometry, new THREE.PointsMaterial({ size: .085, vertexColors: true, transparent: true, opacity: .95, depthWrite: false })));
  scene.add(group);

  return (elapsed) => {
    galaxy.material.rotation = -.3 + Math.sin(elapsed * .08) * .025;
    glow.intensity = 15.5 + Math.sin(elapsed * .35) * 1.5;
  };
}
