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
    for (let i = 0; i < 2400; i += 1) {
      const x = Math.random() * size;
      const y = Math.random() * size;
      const warm = Math.random() < .14;
      context.fillStyle = warm ? `rgba(255,210,175,${.35 + Math.random() * .6})` : `rgba(205,225,255,${.3 + Math.random() * .65})`;
      context.beginPath();
      context.arc(x, y, Math.random() < .018 ? 2.6 : .35 + Math.random() * 1.2, 0, Math.PI * 2);
      context.fill();
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export function addCosmicArchitecture(scene) {
  const group = new THREE.Group();
  const starTexture = cosmicTexture("stars");

  const ceiling = new THREE.Mesh(
    new THREE.PlaneGeometry(16.1, 19.6),
    new THREE.MeshBasicMaterial({ map: starTexture, side: THREE.DoubleSide, transparent: true, opacity: .97 })
  );
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.set(0, 6.18, -.7);
  group.add(ceiling);

  const galaxy = new THREE.Mesh(
    new THREE.PlaneGeometry(10.5, 10.5),
    new THREE.MeshBasicMaterial({ map: cosmicTexture("galaxy"), side: THREE.DoubleSide, transparent: true, opacity: .9, depthWrite: false, blending: THREE.AdditiveBlending })
  );
  galaxy.rotation.x = Math.PI / 2;
  galaxy.rotation.z = -.34;
  galaxy.position.set(0, 6.08, -4.2);
  group.add(galaxy);

  const glow = new THREE.PointLight(0x728dff, 17, 15, 2);
  glow.position.set(0, 5.65, -4.2);
  group.add(glow);

  const hazeMaterial = new THREE.SpriteMaterial({ map: starTexture, color: 0x6878e8, transparent: true, opacity: .07, depthWrite: false, blending: THREE.AdditiveBlending });
  [[-6.7,3.7,-7.8,8,5],[6.8,3.2,-6.2,7,4.5],[-5.8,2.8,2.5,6,4],[6.2,4,1.4,6,4]].forEach(([x,y,z,width,height]) => {
    const cloud = new THREE.Sprite(hazeMaterial.clone());
    cloud.position.set(x, y, z);
    cloud.scale.set(width, height, 1);
    group.add(cloud);
  });

  const positions = [];
  const colors = [];
  const cool = new THREE.Color(0x9acbff);
  const warm = new THREE.Color(0xffd2a0);
  for (let i = 0; i < 1400; i += 1) {
    const side = Math.floor(Math.random() * 3);
    if (side === 0) positions.push(THREE.MathUtils.randFloatSpread(15.5), 5.9 + Math.random() * .16, -10 + Math.random() * 19);
    if (side === 1) positions.push(-8.12 + Math.random() * .08, .5 + Math.random() * 5.2, -10 + Math.random() * 19);
    if (side === 2) positions.push(8.12 - Math.random() * .08, .5 + Math.random() * 5.2, -10 + Math.random() * 19);
    const color = Math.random() < .12 ? warm : cool;
    colors.push(color.r, color.g, color.b);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  group.add(new THREE.Points(geometry, new THREE.PointsMaterial({ size: .042, vertexColors: true, transparent: true, opacity: .95, depthWrite: false })));
  scene.add(group);

  return (elapsed) => {
    galaxy.rotation.z = -.34 + Math.sin(elapsed * .08) * .025;
    glow.intensity = 15.5 + Math.sin(elapsed * .35) * 1.5;
  };
}
