import init, { GraphicsContext } from "boids-3d-rs";
const CANVAS_ID = "canvas";



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
  const handle_slider = (event) => {
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

  count_label.innerHTML = gl_context.get_node_count();

  const update_dimensions = () => {

    const displayWidth = canvas_elem.clientWidth;
    const displayHeight = canvas_elem.clientHeight;

    // Check if the canvas is not the same size.
    const needResize = canvas_elem.width !== displayWidth ||
      canvas_elem.height !== displayHeight;

    if (needResize) {
      // Make the canvas the same size
      canvas_elem.width = displayWidth;
      canvas_elem.height = displayHeight;

      gl_context.set_canvas_dimensions(displayWidth, displayHeight);
    }
  };
  window.addEventListener('resize', () => {
    update_dimensions();
  }, true);

  update_dimensions();

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
  })
  canvas_elem.addEventListener("click", () => {
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
            // Some platforms may not support unadjusted movement.
            // You can request again a regular pointer lock.
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
  fps_label.innerHTML = `${(sum / FPS_FRAME_COUNT).toFixed(2)} ` + "ms/f";
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
