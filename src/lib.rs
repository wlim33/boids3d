mod boids;
mod scene;
mod utils;
use core::f32;
use nalgebra::{Matrix4, UnitQuaternion, Vector3};
use rand::Rng;
use wasm_bindgen::prelude::*;
use web_sys::{HtmlImageElement, WebGl2RenderingContext, WebGlProgram, WebGlTexture};

#[wasm_bindgen]
pub struct GraphicsContext {
    gl: WebGl2RenderingContext,
    program: WebGlProgram,
    height: f32,
    width: f32,
    last_time_stamp: f32,
    max_time_step: f32,
    time_step_accumulator: f32,
    uniforms: GlobalUniforms,
    world: scene::World,
}

#[wasm_bindgen]
impl GraphicsContext {
    pub fn new(canvas_id: &str, height: f32, width: f32) -> Self {
        utils::set_panic_hook();
        let mut scene_world = scene::World::new();
        let gl = utils::new_webgl_context(canvas_id).unwrap();
        //let skybox_program = setup_skybox_shaders(&gl).unwrap();
        let program = utils::setup_shaders(&gl).unwrap();
        //let mut uniform_locs = HashMap::new();

        let shape = RenderShape::Cube;

        let sprite_sheet: HtmlImageElement = utils::get_sprite_sheet("spritesheet").unwrap();
        let texture: WebGlTexture =
            utils::load_image_into_3_d_texture(&gl, &sprite_sheet, 13, 7).unwrap();
        let mut rng = rand::thread_rng();
        //scene_world.generate_bodies(100, rng.gen_range(0..=91));
        scene_world.generate_boids(100, rng.gen_range(0..=91));
        //scene_world.generate_nested_bodies(10);
        load_attributes(&gl, &program, shape);

        let global_uniforms = GlobalUniforms {
            view_projection: Matrix4::identity(),
            view_inverse: Matrix4::identity(),
            light_world_pos: Vector3::new(-50.0, 30.0, 100.0),
            light_color: nalgebra::Vector4::new(1.0, 1.0, 1.0, 1.0),
            texture,
        };

        GraphicsContext {
            gl: gl.clone(),
            program,
            uniforms: global_uniforms,
            height,
            width,
            last_time_stamp: 0.0,
            max_time_step: 1.0 / 60.0,
            time_step_accumulator: 0.0,
            world: scene_world,
        }
    }
    pub fn calculate_view(&mut self) {
        self.uniforms.view_inverse = self.world.get_camera_view_inverse();
        self.uniforms.view_projection =
            self.world.camera_info.perpective.as_matrix() * self.uniforms.view_inverse;
    }

    pub fn set_max_timestep(&mut self, ms: f32) {
        self.max_time_step = ms;
    }

    pub fn get_max_timestep(&self) -> f32 {
        self.max_time_step
    }
    pub fn get_node_count(&self) -> usize {
        self.world.get_node_count()
    }
    pub fn set_boids_params(
        &mut self,
        cohesion: f32,
        turn: f32,
        matching: f32,
        avoid: f32,
        //min_speed: f32,
        //max_speed: f32,
        //visual_range: f32,
        //protected_range: f32,
    ) {
        self.world.bonus_parameters.cohesion_factor = cohesion;
        self.world.bonus_parameters.turn_factor = turn;
        self.world.bonus_parameters.matching_factor = matching;
        self.world.bonus_parameters.avoid_factor = avoid;
        //self.world.bonus_parameters.min_speed = min_speed;
        //self.world.bonus_parameters.max_speed = max_speed;
        //self.world.bonus_parameters.visual_range = visual_range;
        //self.world.bonus_parameters.protected_range = protected_range;
    }

    pub fn add_boids(&mut self, count: usize) {
        let mut rng = rand::thread_rng();
        self.world.generate_boids(count, rng.gen_range(0..=91));
    }

    pub fn raw_tick(&mut self, ms: f32) {
        let t = ms * 0.001;
        let frame_time = t - self.last_time_stamp;
        self.last_time_stamp = t;
        //self.world.update(frame_time);
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
        //self.gl.viewport(0, 0, height as i32, width as i32);
        self.height = height;
        self.width = width;
        self.world.update_aspect_ratio(width, height);
    }

    fn draw(&mut self, render_ids: Vec<scene::NodeID>) {
        self.gl
            .viewport(0, 0, self.width as i32, self.height as i32);
        self.gl.clear_color(0.0, 0.0, 0.0, 0.0);
        self.gl.clear_depth(1.0);
        self.gl.enable(WebGl2RenderingContext::DEPTH_TEST);
        self.gl.enable(WebGl2RenderingContext::CULL_FACE);
        self.gl.depth_func(WebGl2RenderingContext::LEQUAL);
        self.gl.clear(
            WebGl2RenderingContext::COLOR_BUFFER_BIT | WebGl2RenderingContext::DEPTH_BUFFER_BIT,
        );
        for r_id in render_ids {
            draw_object(
                &self.gl,
                &self.program,
                self.world.get_node(r_id),
                self.world.get_materials_node(r_id),
                &self.uniforms,
            )
        }
    }

    pub fn zoom_camera(&mut self, delta: f32) {
        self.world.zoom_camera(delta);
    }

    pub fn move_camera(&mut self, right: f32, up: f32, forward: f32) {
        self.world.move_camera(Vector3::new(right, up, forward));
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
                -0.5, -0.5, 0.0, //0
                0.5, -0.5, 0.0, //1.0
                0.0, 0.5, 0.0, //2
            ];
            let tex_coord: [f32; 6] = [
                0.0, 0.0, //
                0.0, 1.0, //
                1.0, 0.0, //
            ];
            let normals: [f32; 9] = [
                -1.0, -1.0, 0.5, //0
                1.0, -1.0, 0.5, //1.0
                -1.0, 1.0, 0.5, //2
            ];

            let indices: [u16; 3] = [0, 1, 2];
            utils::setup_vertices_indices(
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
                0, 1, 2, // a
                0, 2, 3, // a
                4, 5, 6, // a
                4, 6, 7, // a
                8, 9, 10, // a
                8, 10, 11, // a
                12, 13, 14, // a
                12, 14, 15, // a
                16, 17, 18, // a
                16, 18, 19, // a
                20, 21, 22, // a
                20, 22, 23, // a
            ];
            let face_indices: [u8; 24] = [
                0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
            ];
            utils::setup_attributes_2d_array(
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
                0, 1, 2, // vertex
                1, 2, 3, // vertex
            ];

            utils::setup_vertices_indices(
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

pub fn draw_object(
    gl: &WebGl2RenderingContext,
    shader_program: &WebGlProgram,
    obj: &scene::Node,
    materials: &scene::RenderNode,
    globals: &GlobalUniforms,
) {
    //let world =
    //    Matrix4::from(obj.orientation.to_homogeneous()).append_translation(&obj.world_position);
    let world: Matrix4<f32> = obj.global_transform;
    let world_view_projection: Matrix4<f32> = globals.view_projection * world;
    //world_view_projection[(0, 0)] *= 2.0;
    //world_view_projection[(1, 1)] *= 0.5;
    //world_view_projection[(2, 2)] *= 0.5;
    let world_inverse_transpose = world.transpose();
    utils::write_uniform_mat4(
        gl,
        shader_program,
        world_view_projection.as_slice(),
        "u_worldViewProjection",
    );
    utils::write_uniform_mat4(gl, shader_program, world.as_slice(), "u_world");
    utils::write_uniform_mat4(
        gl,
        shader_program,
        world_inverse_transpose.as_slice(),
        "u_worldInverseTranspose",
    );

    utils::write_uniform_mat4(
        gl,
        shader_program,
        globals.view_inverse.as_slice(),
        "u_viewInverse",
    );
    utils::write_uniform_vec3(
        gl,
        shader_program,
        globals.light_world_pos.as_slice(),
        "u_lightWorldPos",
    );
    utils::write_uniform_vec4(
        gl,
        shader_program,
        globals.light_color.as_slice(),
        "u_lightColor",
    );

    utils::write_uniform_vec4(
        gl,
        shader_program,
        materials.color_mult.as_slice(),
        "u_colorMult",
    );
    utils::write_uniform_vec4(
        gl,
        shader_program,
        materials.specular.as_slice(),
        "u_specular",
    );
    utils::write_uniform_float(gl, shader_program, materials.shininess, "u_shininess");
    utils::write_uniform_float(
        gl,
        shader_program,
        materials.specular_factor,
        "u_specularFactor",
    );
    utils::write_uniform_texture_unit(
        gl,
        shader_program,
        materials.diffuse_texture_unit,
        &globals.texture,
        "u_diffuse",
    );
    utils::write_uniform_uint_array(gl, shader_program, &materials.face_index, "u_faceIndex");
    gl.draw_elements_with_i32(
        WebGl2RenderingContext::TRIANGLES,
        match materials.shape {
            RenderShape::Cube => 36_i32,
            RenderShape::Tetrahedron => 12_i32,
            RenderShape::Quad => 6_i32,
            RenderShape::Triangle => 3_i32,
            _ => 3_i32,
        },
        WebGl2RenderingContext::UNSIGNED_SHORT,
        0,
    );
}
