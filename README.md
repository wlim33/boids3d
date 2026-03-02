# Boids 3D
[![CI](https://github.com/wlim33/boids3d/actions/workflows/main.yml/badge.svg)](https://github.com/wlim33/boids3d/actions/workflows/main.yml)
[![crates.io](https://img.shields.io/crates/v/boids-3d-rs?label=crate)](https://crates.io/crates/boids-3d-rs)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE_APACHE)
[![wasm-pack](https://img.shields.io/badge/wasm--pack-supported-239120?logo=wasm)](https://rustwasm.github.io/wasm-pack/)
Boids is an artificial life program that mimics the flocking behavior of birds. This is an implementation of the original algorithm, extended to 3 dimensions.
**live demo:** [https://boids.wlim.dev/](https://boids.wlim.dev/)

## Description
I started this project to learn more about game engine design and graphics programming. I chose WebGl for its ubiquity, as most people use browsers that are compatible with WebGl 2.0, and with [`wasm-pack`](https://github.com/rustwasm/wasm-pack), developing with Rust in the browser was a pain free experience. Because of the large number of matrix calculations, I figured that the program would benefit from being written in Rust over JavaScript. This decision also opens up the possibility of running the program from the desktop, with minimal rewrites.

My goal is to have a starting point for future graphics projects, without the use of existing engines like Unity or Bevy.

## Runtime dependencies
* A web browser compatible with WebGl 2.0 (check if your browser supports it here: [link](https://get.webgl.org/webgl2/))

## Installing
The program is available as an [npm package](https://www.npmjs.com/package/boids-3d-rs), but is currently unstable.

### Running locally
#### Dependencies
* Rust 1.30.0 or later
* npm 10.9.0 or later
* [wasm-pack](https://github.com/rustwasm/wasm-pack)
* WebGl 2.0 compatible browser

#### Compile the program
Make sure [`wasm-pack`](https://github.com/rustwasm/wasm-pack) is installed by following the directions there, and then from the root directory run:
```
wasm-pack build --target=web
```
Optionally, watch for changes to `.rs` files with [`watchexec`](https://github.com/watchexec/watchexec):
```
 watchexec -e rs wasm-pack build --target=web
```
#### Start the development server
From the `www` directory:
```
npm run start
```
Navigate to [http://localhost:8080/](http://localhost:8080/). You should see some cubes flying around! 

## How It Works:
### Every frame
The compiled program exposes a function that is intended to be called every frame, `raw_tick`, which takes in a raw timestamp. The delta time is then calculated and passed to the scene graph which traverses the nodes and and calculates the transformations for each node in world space for the next frame. The updated matrix values are loaded into the corresponding WebGl buffers and the boids are then drawn.

### Inputs
Camera movement is handled by JavaScript events, and subsequently passed to the program. It is possible to handle the inputs from the Rust program directly, through the use of the [`web-sys`](https://rustwasm.github.io/wasm-bindgen/api/web_sys/) crate, but eventually I want users to have the option of running the program directly, without a browser, where inputs are handled by a lower level library like [`sdl`](https://www.libsdl.org/). Handling the inputs in JavaScript was also easier to manage, as the `web-sys` crate is not as well documented as the browser API it abstracts.

* `scene.rs` contains the scene graph implementation.
* `lib.rs` contains the render pipeline and the exposed functions that are called from the browser.
* `gl_utils.rs` contains the majority of the OpenGl setup and buffer writes. 
* You can drag a `.gltf` file directly onto the canvas (with embedded buffers and 16-bit indices) to swap the boid mesh at runtime.

### UI Controls
The browser UI wraps the wasm module, exposing these convenient controls:

* **Starting boids slider** – change the initial flock size and immediately reset the scene so the renderer starts with the requested count.
* **Parameter sliders (cohesion, matching, avoidance, turn)** – tweak the three steering factors live and watch the behavior update. Labels next to each slider show the current value.
* **FPS readout** – shows the rolling average frame time in milliseconds plus the equivalent FPS (`avg_ms ms | fps`).
* **Add buttons (1/10/100)** – spawn more boids on demand and update the on-screen count.
* **Mouse controls** – left-click to lock the pointer and look around, scroll to adjust FOV, WASD + arrows to move, and `P` to drop extra cubes referenced by the scene graph.

## Next Steps
### Scene Graph
* I'd like to add the option of custom JavaScript hooks for different bodies, as currently the types and behavior of each entity is handled at compile time.
### Rendering
* GPU instancing, as currently each boid has it's own uniforms which means more draw calls. This might require [batching](https://docs.unity3d.com/Manual/DrawCallBatching.html) if I add the option to have different entities to be drawn at the same time.
### Physics Engine
* Boids is pretty simple as the original program just takes in one parameter, delta time. Adding a physics engine will  require a more fleshed out pipeline.

## Testing & Benchmarking
The repo now exposes unit tests for the scene graph as well as a `criterion` benchmark that times `boids_update`.

- `cargo test`
- `cargo bench --bench boids_update`
- `cargo bench --bench boids_update` (runs the full suite covering `boids_update`, `world_update`, render object collection, and `generate_boids` across multiple scales)


## License
This project is licensed under the Apache 2.0 License - see the LICENSE.md file for details

## Acknowledgments
* [wasm-pack](https://github.com/rustwasm/wasm-pack)
* [https://webglfundamentals.org/](https://webglfundamentals.org/)
* [https://nalgebra.org/](https://nalgebra.org/)
