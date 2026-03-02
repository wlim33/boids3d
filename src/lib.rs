mod boids;
mod gl_utils;
mod scene;
use crate::gl_utils::{AttributeLocations, UniformLocations};
use core::f32;
use js_sys::{Float32Array, Uint16Array};
use nalgebra::{Matrix4, UnitQuaternion, Vector3};
use rand::Rng;
pub use scene::World;
use std::collections::HashSet;
use wasm_bindgen::prelude::*;
use web_sys::{HtmlImageElement, WebGl2RenderingContext, WebGlBuffer, WebGlProgram, WebGlTexture};

#[wasm_bindgen]
pub struct GraphicsContext {
    gl: WebGl2RenderingContext,
    attribute_locations: AttributeLocations,
    uniform_locations: UniformLocations,
    program: WebGlProgram,
    height: f32,
    width: f32,
    last_time_stamp: f32,
    max_time_step: f32,
    time_step_accumulator: f32,
    uniforms: GlobalUniforms,
    world: scene::World,
    instancing_enabled: bool,
    instanced_world_buffer: Option<WebGlBuffer>,
    instanced_normal_buffer: Option<WebGlBuffer>,
}

#[wasm_bindgen]
impl GraphicsContext {
    pub fn new(canvas_id: &str, height: f32, width: f32) -> Self {
        gl_utils::set_panic_hook();
        let mut scene_world = scene::World::new();
        let gl = gl_utils::new_webgl_context(canvas_id).unwrap();
        let program = gl_utils::setup_shaders(&gl).unwrap();

        let boids_shape = RenderShape::Cube;
        let sprite_sheet: HtmlImageElement = gl_utils::get_sprite_sheet("spritesheet").unwrap();
        let texture: WebGlTexture =
            gl_utils::load_image_into_3_d_texture(&gl, &sprite_sheet, 13, 7).unwrap();
        let mut rng = rand::thread_rng();
        scene_world.generate_boids(100, rng.gen_range(0..=91));
        load_attributes(&gl, &program, boids_shape);

        let global_uniforms = GlobalUniforms {
            view_projection: Matrix4::identity(),
            view_inverse: Matrix4::identity(),
            light_world_pos: Vector3::new(-50.0, 30.0, 100.0),
            light_color: nalgebra::Vector4::new(1.0, 1.0, 1.0, 1.0),
            texture,
        };

        let attribute_locations = gl_utils::get_attribute_locations(&gl, &program);
        let uniform_locations = gl_utils::get_uniform_locations(&gl, &program);
        GraphicsContext {
            gl: gl.clone(),
            attribute_locations,
            uniform_locations,
            program,
            uniforms: global_uniforms,
            height,
            width,
            last_time_stamp: 0.0,
            max_time_step: 1.0 / 60.0,
            time_step_accumulator: 0.0,
            world: scene_world,
            instancing_enabled: false,
            instanced_world_buffer: None,
            instanced_normal_buffer: None,
        }
    }
    pub fn calculate_view(&mut self) {
        self.uniforms.view_inverse = self.world.get_camera_view_inverse();
        self.uniforms.view_projection =
            self.world.camera_info.perpective.as_matrix() * self.uniforms.view_inverse;
    }

    fn write_use_instance_matrices(&self, enabled: bool) {
        self.gl.uniform1i(
            Some(&self.uniform_locations.use_instance_matrices),
            enabled as i32,
        );
    }

    fn ensure_instanced_buffer(&mut self, buffer_slot: &mut Option<WebGlBuffer>) -> WebGlBuffer {
        if buffer_slot.is_none() {
            *buffer_slot = Some(self.gl.create_buffer().unwrap());
        }
        buffer_slot.as_ref().unwrap().clone()
    }

    pub fn set_max_timestep(&mut self, ms: f32) {
        self.max_time_step = ms;
    }

    pub fn get_max_timestep(&self) -> f32 {
        self.max_time_step
    }
    pub fn set_instancing_enabled(&mut self, enabled: bool) {
        self.instancing_enabled = enabled;
    }
    pub fn get_node_count(&self) -> usize {
        self.world.get_node_count()
    }
    pub fn set_boids_params(&mut self, cohesion: f32, turn: f32, matching: f32, avoid: f32) {
        self.world.boids_params.cohesion_factor = cohesion;
        self.world.boids_params.turn_factor = turn;
        self.world.boids_params.matching_factor = matching;
        self.world.boids_params.avoid_factor = avoid;
    }

    pub fn add_boids(&mut self, count: usize) {
        let mut rng = rand::thread_rng();
        self.world.generate_boids(count, rng.gen_range(0..=91));
    }

    pub fn reset_boids(&mut self, count: usize) {
        let mut rng = rand::thread_rng();
        self.world = scene::World::new();
        self.world.generate_boids(count, rng.gen_range(0..=91));
        self.calculate_view();
        self.last_time_stamp = 0.0;
        self.time_step_accumulator = 0.0;
    }

    pub fn raw_tick(&mut self, ms: f32) {
        let t = ms * 0.001;
        let frame_time = t - self.last_time_stamp;
        self.last_time_stamp = t;
        self.world.boids_update(frame_time);
        self.time_step_accumulator += frame_time;
        self.calculate_view();
        let render_ids = self.world.calculate_final_render_objects();
        self.draw(render_ids);
    }

    pub fn rotate_camera(&mut self, dx: f32, dy: f32) {
        self.world.rotate_camera(dx, dy);
    }

    pub fn add_cube(&mut self) {
        self.world.add_node_at_view();
    }

    pub fn set_canvas_dimensions(&mut self, width: f32, height: f32) {
        self.height = height;
        self.width = width;
        self.world.update_aspect_ratio(width, height);
    }

    pub fn replace_boid_mesh(
        &mut self,
        positions: Float32Array,
        normals: Float32Array,
        tex_coords: Float32Array,
        indices: Uint16Array,
    ) -> Result<(), JsValue> {
        let vertex_data = float32_array_to_vec(&positions);
        if vertex_data.len() % 3 != 0 {
            return Err(JsValue::from_str(
                "positions must contain 3 floats per vertex",
            ));
        }
        if vertex_data.is_empty() {
            return Err(JsValue::from_str("positions cannot be empty"));
        }
        let vertex_count = vertex_data.len() / 3;
        let normals_vec = ensure_length(float32_array_to_vec(&normals), vertex_data.len());
        let tex_coords_len = vertex_count * 2;
        let tex_coords_vec = ensure_length(float32_array_to_vec(&tex_coords), tex_coords_len);

        let mut index_data = vec![0u16; indices.length() as usize];
        indices.copy_to(&mut index_data);
        if index_data.is_empty() {
            return Err(JsValue::from_str("mesh must define at least one triangle"));
        }

        let face_indices = vec![0u8; vertex_count];
        load_custom_mesh(
            &self.gl,
            &self.program,
            &vertex_data,
            &tex_coords_vec,
            &normals_vec,
            &index_data,
            &face_indices,
        );
        self.world.set_global_index_count(index_data.len() as i32);
        Ok(())
    }

    fn draw_instanced(&mut self, transforms: &[Matrix4<f32>], materials: &scene::RenderNode) {
        if transforms.is_empty() {
            return;
        }
        self.write_instanced_matrices(transforms);
        self.write_use_instance_matrices(true);
        let identity = Matrix4::identity();
        self.write_uniform_mat4(identity.as_slice(), &self.uniform_locations.world);
        self.write_uniform_mat4(
            identity.as_slice(),
            &self.uniform_locations.world_inverse_transpose,
        );
        self.write_uniform_uint_array(&materials.face_index, &self.uniform_locations.face_index);
        self.write_uniform_texture_unit(
            materials.diffuse_texture_unit,
            &self.uniforms.texture,
            &self.uniform_locations.diffuse,
        );
        self.gl.draw_elements_instanced_with_i32(
            WebGl2RenderingContext::TRIANGLES,
            materials.index_count,
            WebGl2RenderingContext::UNSIGNED_SHORT,
            0,
            transforms.len() as i32,
        );
        self.write_use_instance_matrices(false);
        self.cleanup_instanced_attributes(
            [
                self.attribute_locations.instance_matrix0,
                self.attribute_locations.instance_matrix1,
                self.attribute_locations.instance_matrix2,
                self.attribute_locations.instance_matrix3,
            ],
            [
                self.attribute_locations.instance_normal_matrix0,
                self.attribute_locations.instance_normal_matrix1,
                self.attribute_locations.instance_normal_matrix2,
                self.attribute_locations.instance_normal_matrix3,
            ],
        );
    }

    fn write_instanced_matrices(&mut self, transforms: &[Matrix4<f32>]) {
        let mut world_data = Vec::with_capacity(transforms.len() * 16);
        let mut normal_data = Vec::with_capacity(transforms.len() * 16);
        for transform in transforms {
            world_data.extend_from_slice(transform.as_slice());
            let normal = transform
                .try_inverse()
                .unwrap_or_else(|| Matrix4::identity())
                .transpose();
            normal_data.extend_from_slice(normal.as_slice());
        }
        let world_buffer = self.ensure_instanced_buffer(&mut self.instanced_world_buffer);
        let normal_buffer = self.ensure_instanced_buffer(&mut self.instanced_normal_buffer);
        self.set_instanced_matrix_attributes(
            &world_data,
            [
                self.attribute_locations.instance_matrix0,
                self.attribute_locations.instance_matrix1,
                self.attribute_locations.instance_matrix2,
                self.attribute_locations.instance_matrix3,
            ],
            &world_buffer,
        );
        self.set_instanced_matrix_attributes(
            &normal_data,
            [
                self.attribute_locations.instance_normal_matrix0,
                self.attribute_locations.instance_normal_matrix1,
                self.attribute_locations.instance_normal_matrix2,
                self.attribute_locations.instance_normal_matrix3,
            ],
            &normal_buffer,
        );
    }

    fn set_instanced_matrix_attributes(
        &self,
        data: &[f32],
        locations: [i32; 4],
        buffer: &WebGlBuffer,
    ) {
        let array = unsafe { Float32Array::view(data) };
        self.gl
            .bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(buffer));
        self.gl.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &array,
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
        let bytes_per_float = std::mem::size_of::<f32>() as i32;
        let stride = 16 * bytes_per_float;
        for (i, &loc) in locations.iter().enumerate() {
            if loc < 0 {
                continue;
            }
            let offset = (i as i32) * 4 * bytes_per_float;
            self.gl.enable_vertex_attrib_array(loc as u32);
            self.gl.vertex_attrib_pointer_with_i32(
                loc as u32,
                4,
                WebGl2RenderingContext::FLOAT,
                false,
                stride,
                offset,
            );
            self.gl.vertex_attrib_divisor(loc as u32, 1);
        }
    }

    fn cleanup_instanced_attributes(&self, world_locations: [i32; 4], normal_locations: [i32; 4]) {
        for &loc in world_locations.iter().chain(normal_locations.iter()) {
            if loc < 0 {
                continue;
            }
            self.gl.vertex_attrib_divisor(loc as u32, 0);
            self.gl.disable_vertex_attrib_array(loc as u32);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderShape {
    Triangle,
    Cube,
    Sphere,
    Quad,
    Tetrahedron,
    None,
}

#[wasm_bindgen]
#[derive(Debug, Clone, PartialEq)]
pub struct RenderObject {
    shape: RenderShape,
    world_position: Vector3<f32>,
    scaling: Vector3<f32>,
    orientation: UnitQuaternion<f32>,
}

pub struct GlobalUniforms {
    view_projection: Matrix4<f32>,
    view_inverse: Matrix4<f32>,
    light_world_pos: Vector3<f32>,
    light_color: nalgebra::Vector4<f32>,
    texture: WebGlTexture,
}

pub fn load_attributes(
    gl: &WebGl2RenderingContext,
    shader_program: &WebGlProgram,
    shape: RenderShape,
) {
    match shape {
        RenderShape::Triangle => {
            let vertices: [f32; 9] = [
                -0.5, -0.5, 0.0, // commented to avoid linter (each row is a vertex)
                0.5, -0.5, 0.0, //
                0.0, 0.5, 0.0, //
            ];
            let tex_coord: [f32; 6] = [
                0.0, 0.0, //
                0.0, 1.0, //
                1.0, 0.0, //
            ];
            let normals: [f32; 9] = [
                -1.0, -1.0, 0.5, //
                1.0, -1.0, 0.5, //
                -1.0, 1.0, 0.5, //
            ];

            let indices: [u16; 3] = [0, 1, 2];
            gl_utils::setup_vertices_indices(
                gl,
                &vertices,
                &tex_coord,
                &normals,
                &indices,
                shader_program,
            );
        }
        RenderShape::Cube => {
            let vertices: [f32; 72] = [
                1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
                -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
                -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
                -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            ];
            let tex_coord: [f32; 48] = [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            ];
            let normals: [f32; 72] = [
                1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
                0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0,
                0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
            ];
            let indices: [u16; 36] = [
                0, 1, 2, //
                0, 2, 3, //
                4, 5, 6, //
                4, 6, 7, //
                8, 9, 10, //
                8, 10, 11, //
                12, 13, 14, //
                12, 14, 15, //
                16, 17, 18, //
                16, 18, 19, //
                20, 21, 22, //
                20, 22, 23, //
            ];
            let face_indices: [u8; 24] = [
                0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
            ];
            gl_utils::setup_attributes_2d_array(
                gl,
                &vertices,
                &tex_coord,
                &normals,
                &indices,
                &face_indices,
                shader_program,
            );
        }
        RenderShape::Quad => {
            let vertices: [f32; 12] = [
                0.0, 0.0, 0.0, //
                10.0, 0.0, 0.0, //
                0.0, 10.0, 0.0, //
                10.0, 10.0, 0.0, //
            ];
            let tex_coord: [f32; 8] = [
                0.0, 0.0, //
                0.0, 1.0, //
                1.0, 0.0, //
                1.0, 1.0, //
            ];
            let normals: [f32; 12] = [
                0.0, 0.0, 1.0, //
                0.0, 0.0, 1.0, //
                0.0, 0.0, 1.0, //
                0.0, 0.0, 1.0, //
            ];

            let indices: [u16; 6] = [
                0, 1, 2, //
                1, 2, 3, //
            ];

            gl_utils::setup_vertices_indices(
                gl,
                &vertices,
                &tex_coord,
                &normals,
                &indices,
                shader_program,
            );
        }
        _ => {}
    }
}

pub fn load_custom_mesh(
    gl: &WebGl2RenderingContext,
    shader_program: &WebGlProgram,
    vertices: &[f32],
    tex_coords: &[f32],
    normals: &[f32],
    indices: &[u16],
    face_indices: &[u8],
) {
    gl_utils::setup_attributes_2d_array(
        gl,
        vertices,
        tex_coords,
        normals,
        indices,
        face_indices,
        shader_program,
    );
}

fn float32_array_to_vec(array: &Float32Array) -> Vec<f32> {
    let mut dst = vec![0.0; array.length() as usize];
    array.copy_to(&mut dst);
    dst
}

fn ensure_length(mut data: Vec<f32>, target_len: usize) -> Vec<f32> {
    if data.len() > target_len {
        data.truncate(target_len);
    } else {
        data.resize(target_len, 0.0);
    }
    data
}

pub fn draw_object(
    render_context: &GraphicsContext,
    obj: &scene::Node,
    materials: &scene::RenderNode,
    globals: &GlobalUniforms,
    uniform_locations: &UniformLocations,
) {
    let world: Matrix4<f32> = obj.global_transform;
    let world_inverse_transpose = world.transpose();
    render_context.write_uniform_mat4(world.as_slice(), &uniform_locations.world);
    render_context.write_uniform_mat4(
        world_inverse_transpose.as_slice(),
        &uniform_locations.world_inverse_transpose,
    );
    render_context.write_uniform_vec3(
        globals.light_world_pos.as_slice(),
        &uniform_locations.light_world_pos,
    );
    render_context.write_uniform_vec4(
        globals.light_color.as_slice(),
        &uniform_locations.light_color,
    );

    render_context.write_uniform_texture_unit(
        materials.diffuse_texture_unit,
        &globals.texture,
        &uniform_locations.diffuse,
    );
    render_context.write_uniform_uint_array(&materials.face_index, &uniform_locations.face_index);
    render_context.gl.draw_elements_with_i32(
        WebGl2RenderingContext::TRIANGLES,
        materials.index_count,
        WebGl2RenderingContext::UNSIGNED_SHORT,
        0,
    );
}
