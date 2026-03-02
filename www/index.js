import init, { GraphicsContext } from "boids-3d-rs";
const CANVAS_ID = "canvas";

const isTouchDevice = () => 'ontouchstart' in window || navigator.maxTouchPoints > 0;

function enableDrag(element, drag_handle_element) {
  var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
  if (drag_handle_element) {
    drag_handle_element.onmousedown = dragMouseDown;
  } else {
    element.onmousedown = dragMouseDown;
  }

  function dragMouseDown(e) {
    e.preventDefault();
    pos3 = e.clientX;
    pos4 = e.clientY;
    document.onmouseup = closeDragElement;
    document.onmousemove = elementDrag;
  }

  function elementDrag(e) {
    e.preventDefault();
    pos1 = pos3 - e.clientX;
    pos2 = pos4 - e.clientY;
    pos3 = e.clientX;
    pos4 = e.clientY;
    element.style.top = (element.offsetTop - pos2) + "px";
    element.style.left = (element.offsetLeft - pos1) + "px";
  }

  function closeDragElement() {
    document.onmouseup = null;
    document.onmousemove = null;
  }
}

const COMPONENT_BYTE_SIZE = {
  5120: 1,
  5121: 1,
  5122: 2,
  5123: 2,
  5125: 4,
  5126: 4,
};

function decodeBuffer(bufferDef) {
  if (!bufferDef || !bufferDef.uri) {
    throw new Error('glTF buffers must embed data via a data URI');
  }
  if (!bufferDef.uri.startsWith('data:')) {
    throw new Error('External buffer files are not supported for drag-and-drop');
  }
  const [, base64] = bufferDef.uri.split(',');
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function readFloatAccessor(json, bufferBytes, accessorIndex, componentCount) {
  const accessor = json.accessors?.[accessorIndex];
  if (!accessor) {
    throw new Error('Missing accessor definition for mesh attribute');
  }
  if (accessor.componentType !== 5126) {
    throw new Error('Only float attributes are supported for positions/normals/texcoords');
  }
  if (!accessor.type.startsWith('VEC')) {
    throw new Error('Unexpected accessor type for attribute');
  }
  const elementCount = accessor.count * componentCount;
  const bufferView = json.bufferViews?.[accessor.bufferView];
  if (!bufferView) {
    throw new Error('Accessor does not reference a bufferView');
  }
  const byteOffset = (bufferView.byteOffset || 0) + (accessor.byteOffset || 0);
  const stride = bufferView.byteStride || componentCount * 4;
  if (stride === componentCount * 4) {
    return new Float32Array(bufferBytes.buffer, byteOffset, elementCount);
  }
  const result = new Float32Array(elementCount);
  const view = new DataView(bufferBytes.buffer, byteOffset, accessor.count * stride);
  for (let i = 0; i < accessor.count; i += 1) {
    for (let j = 0; j < componentCount; j += 1) {
      result[i * componentCount + j] = view.getFloat32(i * stride + j * 4, true);
    }
  }
  return result;
}

function readIndices(json, bufferBytes, accessorIndex) {
  const accessor = json.accessors?.[accessorIndex];
  if (!accessor) {
    throw new Error('Missing indices accessor');
  }
  const bufferView = json.bufferViews?.[accessor.bufferView];
  if (!bufferView) {
    throw new Error('Indices accessor has no bufferView');
  }
  const componentSize = COMPONENT_BYTE_SIZE[accessor.componentType];
  if (!componentSize) {
    throw new Error('Unsupported index component type');
  }
  const byteOffset = (bufferView.byteOffset || 0) + (accessor.byteOffset || 0);
  const stride = bufferView.byteStride || componentSize;
  if (stride !== componentSize) {
    throw new Error('Indices with explicit stride are not supported');
  }
  if (accessor.componentType === 5123) {
    return new Uint16Array(bufferBytes.buffer, byteOffset, accessor.count);
  }
  if (accessor.componentType === 5125) {
    const raw = new Uint32Array(bufferBytes.buffer, byteOffset, accessor.count);
    const converted = new Uint16Array(accessor.count);
    for (let i = 0; i < accessor.count; i += 1) {
      if (raw[i] > 0xffff) {
        throw new Error('Meshes with more than 65535 vertices are not supported yet');
      }
      converted[i] = raw[i];
    }
    return converted;
  }
  throw new Error('Unsupported index component type');
}

async function loadGltfMesh(file) {
  const json = JSON.parse(await file.text());
  const bufferDef = json.buffers?.[0];
  if (!bufferDef) {
    throw new Error('glTF file does not define any buffer data');
  }
  const bufferBytes = decodeBuffer(bufferDef);
  const mesh = json.meshes?.[0];
  const primitive = mesh?.primitives?.[0];
  if (!primitive) {
    throw new Error('No mesh primitive found in the glTF file');
  }
  const positionAccessor = primitive.attributes?.POSITION;
  const normalAccessor = primitive.attributes?.NORMAL;
  const texCoordAccessor = primitive.attributes?.TEXCOORD_0;
  if (positionAccessor === undefined || primitive.indices === undefined) {
    throw new Error('Mesh must include positions and indices');
  }
  const positions = readFloatAccessor(json, bufferBytes, positionAccessor, 3);
  const normals = normalAccessor !== undefined
    ? readFloatAccessor(json, bufferBytes, normalAccessor, 3)
    : new Float32Array(positions.length);
  const texCoords = texCoordAccessor !== undefined
    ? readFloatAccessor(json, bufferBytes, texCoordAccessor, 2)
    : new Float32Array((positions.length / 3) * 2);
  const indices = readIndices(json, bufferBytes, primitive.indices);
  return { positions, normals, texCoords, indices };
}


function setup() {
  let drag_section_elem = document.querySelector(`section`);
  let drag_handle_elem = document.querySelector(`#drag-handle`);
  enableDrag(drag_section_elem, drag_handle_elem);

  const canvas_elem = document.querySelector(`#${CANVAS_ID}`);
  const gl_context = GraphicsContext.new(CANVAS_ID, canvas_elem.clientHeight, canvas_elem.clientWidth);
  const cohesion_slider = document.querySelector('#cohesion');
  const matching_slider = document.querySelector('#matching');
  const avoid_slider = document.querySelector('#avoidance');
  const turn_slider = document.querySelector('#turn');

  const cohesion_label = document.querySelector('#cohesion-value');
  const matching_label = document.querySelector('#matching-value');
  const avoid_label = document.querySelector('#avoidance-value');
  const turn_label = document.querySelector('#turn-value');
  const starting_boids_slider = document.querySelector('#starting-boids');
  const starting_boids_label = document.querySelector('#starting-boids-value');


  const add_1_button = document.querySelector('#add-1');
  const add_10_button = document.querySelector('#add-10');
  const add_100_button = document.querySelector('#add-100');
  const count_label = document.querySelector('#count-value');

  add_1_button.addEventListener("click", () => {
    gl_context.add_boids(1);
    count_label.innerHTML = gl_context.get_node_count();
  });

  add_10_button.addEventListener("click", () => {
    gl_context.add_boids(10);
    count_label.innerHTML = gl_context.get_node_count();
  });
  add_100_button.addEventListener("click", () => {
    gl_context.add_boids(100);
    count_label.innerHTML = gl_context.get_node_count();
  });
  const handle_slider = () => {
    cohesion_label.innerHTML = cohesion_slider.value;
    matching_label.innerHTML = matching_slider.value;
    avoid_label.innerHTML = avoid_slider.value;
    turn_label.innerHTML = turn_slider.value;
    gl_context.set_boids_params(cohesion_slider.value, turn_slider.value, matching_slider.value, avoid_slider.value);
  };

  cohesion_slider.addEventListener("change", handle_slider);
  matching_slider.addEventListener("change", handle_slider);
  avoid_slider.addEventListener("change", handle_slider);
  turn_slider.addEventListener("change", handle_slider);

  const update_starting_boids_label = () => {
    if (!starting_boids_slider || !starting_boids_label) return;
    starting_boids_label.innerHTML = starting_boids_slider.value;
  };

  const apply_starting_boids = () => {
    if (!starting_boids_slider) return;
    const desired_count = Number(starting_boids_slider.value);
    update_starting_boids_label();
    gl_context.reset_boids(desired_count);
    if (count_label) {
      count_label.innerHTML = gl_context.get_node_count();
    }
  };

  starting_boids_slider?.addEventListener("input", update_starting_boids_label);
  starting_boids_slider?.addEventListener("change", apply_starting_boids);

  // Mobile: cap boid count and start with fewer boids
  if (isTouchDevice() && starting_boids_slider) {
    starting_boids_slider.max = 300;
    starting_boids_slider.value = 50;
  }

  if (starting_boids_slider) {
    apply_starting_boids();
  } else if (count_label) {
    count_label.innerHTML = gl_context.get_node_count();
  }

  const instancing_checkbox = document.querySelector('#instancing');
  instancing_checkbox?.addEventListener('change', () => {
    gl_context.set_instancing_enabled(instancing_checkbox.checked);
  });
  if (instancing_checkbox) {
    gl_context.set_instancing_enabled(instancing_checkbox.checked);
  }

  const update_dimensions = () => {
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = canvas_elem.clientWidth;
    const displayHeight = canvas_elem.clientHeight;

    const bufferWidth = displayWidth * dpr;
    const bufferHeight = displayHeight * dpr;

    const needResize = canvas_elem.width !== bufferWidth ||
      canvas_elem.height !== bufferHeight;

    if (needResize) {
      canvas_elem.width = bufferWidth;
      canvas_elem.height = bufferHeight;
      gl_context.set_canvas_dimensions(bufferWidth, bufferHeight);
    }
  };
  window.addEventListener('resize', () => {
    update_dimensions();
  }, true);

  // Handle orientation changes on mobile (iOS edge case)
  window.addEventListener('orientationchange', () => {
    setTimeout(update_dimensions, 100);
  });

  update_dimensions();

  // -- Collapsible panel toggle --
  const panelToggle = document.getElementById('panel-toggle');
  const panel = document.querySelector('section');
  if (isTouchDevice() && panel) {
    panel.classList.add('collapsed');
  }
  panelToggle?.addEventListener('click', (e) => {
    e.stopPropagation();
    panel.classList.toggle('collapsed');
  });

  // -- Drag-and-drop GLTF loading --
  const drop_zone = document.querySelector('#drop-zone');
  const drop_status = document.querySelector('#drop-status');
  const set_drop_status = (msg, error = false) => {
    if (!drop_status) return;
    drop_status.textContent = msg;
    drop_status.style.color = error ? '#ff7d7d' : '#9fccff';
  };
  set_drop_status('Drop a .gltf file to replace the boid mesh');

  const handle_drag_highlight = (active) => () => {
    if (!drop_zone) return;
    drop_zone.classList.toggle('active', active);
  };
  document.addEventListener('dragenter', handle_drag_highlight(true));
  document.addEventListener('dragleave', handle_drag_highlight(false));

  const handleGltfFile = async (gltf_file) => {
    set_drop_status('Loading mesh...');
    try {
      const mesh = await loadGltfMesh(gltf_file);
      gl_context.replace_boid_mesh(mesh.positions, mesh.normals, mesh.texCoords, mesh.indices);
      set_drop_status('Custom mesh loaded successfully.');
    } catch (err) {
      console.error(err);
      set_drop_status(err.message || 'Unable to parse glTF mesh', true);
    }
  };

  document.addEventListener('drop', async (event) => {
    handle_drag_highlight(false)();
    const files = event.dataTransfer ? Array.from(event.dataTransfer.files) : [];
    const gltf_file = files.find((file) => file.name.toLowerCase().endsWith('.gltf'));
    if (!gltf_file) {
      set_drop_status('Drop a .gltf file to swap the mesh', true);
      return;
    }
    await handleGltfFile(gltf_file);
  });

  // -- File input for GLTF on mobile --
  const gltfFileInput = document.getElementById('gltf-file');
  gltfFileInput?.addEventListener('change', async () => {
    const file = gltfFileInput.files?.[0];
    if (file) {
      await handleGltfFile(file);
      gltfFileInput.value = '';
    }
  });

  // -- Mouse camera controls (desktop) --
  document.addEventListener("mousemove", (event) => {
    if (document.pointerLockElement !== canvas_elem) {
      return;
    }
    gl_context.rotate_camera(event.movementX, event.movementY);
  });

  document.addEventListener('keydown', (event) => {
    let x = 0.0;
    let y = 0.0;
    let z = 0.0;
    let pitch = 0.0;
    let yaw = 0.0;
    switch (event.code) {

      case "KeyP":
        gl_context.add_cube();
        break;
      case "KeyA":
      case "ArrowLeft":
        x = -1.0;
        break;
      case "KeyD":
      case "ArrowRight":
        x = 1.0;
        break;
      //case "KeyW":
      //  y = 1.0;
      //  break;
      //case "KeyR":
      //  y = -1.0;
      //  break;
      case "ArrowUp":
      case "KeyW":
        z = 1.0;
        break;
      case "ArrowDown":
      case "KeyS":
        z = -1.0;
        break;
    }
    gl_context.move_camera(x, y, z);
    if (yaw !== 0.0 || pitch !== 0.0) {
      gl_context.rotate_camera(yaw, pitch);
    }
  });
  document.addEventListener("wheel", (event) => {
    gl_context.zoom_camera(event.deltaY);
  });

  // -- Touch camera controls (mobile) --
  let lastTouch = null;
  let lastPinchDist = null;

  function handlePinch(e) {
    const t0 = e.touches[0], t1 = e.touches[1];
    const dist = Math.hypot(t1.clientX - t0.clientX, t1.clientY - t0.clientY);
    if (lastPinchDist !== null) {
      const delta = lastPinchDist - dist;
      gl_context.zoom_camera(delta * 2);
    }
    lastPinchDist = dist;
  }

  canvas_elem.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) {
      lastTouch = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }
    e.preventDefault();
  }, { passive: false });

  canvas_elem.addEventListener('touchmove', (e) => {
    if (e.touches.length === 1 && lastTouch) {
      const dx = e.touches[0].clientX - lastTouch.x;
      const dy = e.touches[0].clientY - lastTouch.y;
      gl_context.rotate_camera(dx, dy);
      lastTouch = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }
    if (e.touches.length === 2) {
      handlePinch(e);
    }
    e.preventDefault();
  }, { passive: false });

  canvas_elem.addEventListener('touchend', (e) => {
    lastTouch = null;
    lastPinchDist = null;
    e.preventDefault();
  }, { passive: false });

  // Prevent pull-to-refresh / rubber-banding on canvas
  document.addEventListener('touchmove', (e) => {
    if (e.target === canvas_elem) e.preventDefault();
  }, { passive: false });

  // -- Pointer lock (desktop only) --
  canvas_elem.addEventListener("click", () => {
    if (isTouchDevice()) return;
    if (document.pointerLockElement === canvas_elem) {
      document.exitPointerLock();
    } else {
      const promise = canvas_elem.requestPointerLock({
        unadjustedMovement: true,
      });

      if (!promise) {
        console.log("disabling mouse acceleration is not supported");
        return;
      }

      return promise
        .then(() => {
          console.log("pointer is locked")
        })
        .catch((error) => {
          if (error.name === "NotSupportedError") {
            return canvas_elem.requestPointerLock();
          }
        });
    }
  });
  return gl_context;
}

const fps_label = document.querySelector('#fps-value');
const FPS_FRAME_COUNT = 100;
let frame_times = new Array(FPS_FRAME_COUNT).fill(0.0);
let sum = 0;
let right = 0;
const fps_moving_average = (frame_ms) => {
  right = (right + 1) % FPS_FRAME_COUNT;
  sum -= frame_times[right];
  frame_times[right] = frame_ms;
  sum += frame_ms;
  const avg_ms = sum / FPS_FRAME_COUNT;
  const fps = avg_ms > 0 ? 1000 / avg_ms : 0;
  fps_label.innerHTML = `${avg_ms.toFixed(2)} ms | ${fps.toFixed(1)} fps`;
}

async function run() {
  await init();
  const universe = setup();
  let t0 = performance.now();
  function draw() {
    universe.raw_tick(performance.now());
    requestAnimationFrame(draw);
    const t1 = performance.now();
    fps_moving_average((t1 - t0));
    t0 = t1;
  }
  requestAnimationFrame(draw);
}
run();
