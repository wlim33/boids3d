pub mod kdtree;
pub mod render3d;
mod utils;
use core::f32;
use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3};
use render3d::{Boid, BoidUniverse};
use utils::{get_sprite_coordinates, update_bodies};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlImageElement, WebGl2RenderingContext, WebGlProgram, WebGlShader, WebGlTexture};

pub fn init_webgl_context(canvas_id: &str) -> Result<WebGl2RenderingContext, JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id(canvas_id).unwrap();

    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;

    let gl: WebGl2RenderingContext = canvas
        .get_context("webgl2")?
        .unwrap()
        .dyn_into::<WebGl2RenderingContext>()
        .unwrap();
    gl.viewport(0, 0, canvas.width() as i32, canvas.height() as i32);
    log!(
        "viewport({}, {}, {}, {})",
        0,
        0,
        canvas.width() as i32,
        canvas.height() as i32
    );
    Ok(gl)
}

pub fn create_shader(
    gl: &WebGl2RenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, JsValue> {
    let shader = gl
        .create_shader(shader_type)
        .ok_or_else(|| JsValue::from_str("Unable to create shader obj"))?;
    gl.shader_source(&shader, source);

    gl.compile_shader(&shader);

    if gl
        .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(JsValue::from_str(
            &gl.get_shader_info_log(&shader)
                .unwrap_or_else(|| "Unknown error creating shader".into()),
        ))
    }
}

pub fn setup_shaders(gl: &WebGl2RenderingContext) -> Result<WebGlProgram, JsValue> {
    let vertex_shader = create_shader(
        gl,
        WebGl2RenderingContext::VERTEX_SHADER,
        r##"#version 300 es

        uniform mat4 u_worldViewProjection;
        uniform mat4 u_world;
        uniform mat4 u_viewInverse;
        uniform mat4 u_worldInverseTranspose;
        uniform vec3 u_lightWorldPos;

        in vec4 position;
        in vec3 normal;
        in vec2 texcoord;
        in uint a_faceId;

        out vec4 v_position;
        out vec2 v_texCoord;
        out vec3 v_normal;
        out vec3 v_surfaceToLight;
        out vec3 v_surfaceToView;
        flat out uint v_faceId;

        void main() {
            v_faceId = a_faceId;
            v_texCoord = texcoord;
            v_position = u_worldViewProjection * position;
            v_normal = (u_worldInverseTranspose * vec4(normal, 0)).xyz;
            v_surfaceToLight = u_lightWorldPos - (u_world * position).xyz;
            v_surfaceToView = (u_viewInverse[3] - (u_world * position)).xyz;
            gl_Position = v_position;
        }
        "##,
    )
    .unwrap();

    let fragment_shader = create_shader(
        gl,
        WebGl2RenderingContext::FRAGMENT_SHADER,
        r##"#version 300 es
        precision highp float;

        in vec4 v_position;
        in vec2 v_texCoord;
        in vec3 v_normal;
        in vec3 v_surfaceToLight;
        in vec3 v_surfaceToView;
        flat in uint v_faceId;

        uniform vec4 u_lightColor;
        uniform vec4 u_colorMult;
        uniform mediump sampler2DArray u_diffuse;
        uniform uint u_faceIndex[6];
        uniform vec4 u_specular;
        uniform float u_shininess;
        uniform float u_specularFactor;

        out vec4 color;
        vec4 lit(float l ,float h, float m) {
            return vec4(1.0, abs(l), (l > 0.0) ? pow(max(0.0, h), m) : 0.0, 1.0);
        }

        void main() {
          vec4 diffuseColor = texture(u_diffuse, vec3(v_texCoord, u_faceIndex[v_faceId]));;
          vec3 a_normal = normalize(v_normal);
          vec3 surfaceToLight = normalize(v_surfaceToLight);
          vec3 surfaceToView = normalize(v_surfaceToView);
          vec3 halfVector = normalize(v_surfaceToLight + v_surfaceToView);
          vec4 litR = lit(dot(a_normal, v_surfaceToLight), dot(a_normal, halfVector), u_shininess);
          //color = vec4((u_lightColor * (u_diffuseColor * litR.y * u_colorMult + u_specular * litR.z * u_specularFactor)).rgb, diffuseColor.a);
          color = diffuseColor;
          //color = vec4(0.0, 0.0, 1.0, 1.0);
        }
        "##,
    )
    .unwrap();

    let shader_program = gl.create_program().unwrap();
    gl.attach_shader(&shader_program, &vertex_shader);
    gl.attach_shader(&shader_program, &fragment_shader);
    gl.link_program(&shader_program);

    if gl
        .get_program_parameter(&shader_program, WebGl2RenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        gl.use_program(Some(&shader_program));
        Ok(shader_program)
    } else {
        Err(JsValue::from_str(
            &gl.get_program_info_log(&shader_program)
                .unwrap_or_else(|| "Unknown error linking program.".into()),
        ))
    }
}

pub fn setup_attributes_2d_array(
    gl: &WebGl2RenderingContext,
    vertices: &[f32],
    tex_coords: &[f32],
    normals: &[f32],
    indices: &[u16],
    face_indices: &[u32],
    shader_program: &WebGlProgram,
) {
    let indices_array = unsafe { js_sys::Uint16Array::view(indices) };
    let indices_buffer = gl.create_buffer().unwrap();
    gl.bind_buffer(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        Some(&indices_buffer),
    );

    gl.buffer_data_with_array_buffer_view(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        &indices_array,
        WebGl2RenderingContext::DYNAMIC_DRAW,
    );
    utils::set_float_buffer_attribute(gl, vertices, 3, "position", shader_program);
    utils::set_float_buffer_attribute(gl, normals, 3, "normal", shader_program);
    utils::set_float_buffer_attribute(gl, tex_coords, 2, "texcoord", shader_program);
    utils::set_int_buffer_attribute(gl, face_indices, 1, "a_faceId", shader_program);
}
pub fn setup_vertices_indices(
    gl: &WebGl2RenderingContext,
    vertices: &[f32],
    tex_coords: &[f32],
    normals: &[f32],
    indices: &[u16],
    shader_program: &WebGlProgram,
) {
    let indices_array = unsafe { js_sys::Uint16Array::view(indices) };
    let indices_buffer = gl.create_buffer().unwrap();
    gl.bind_buffer(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        Some(&indices_buffer),
    );

    gl.buffer_data_with_array_buffer_view(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        &indices_array,
        WebGl2RenderingContext::DYNAMIC_DRAW,
    );
    utils::set_float_buffer_attribute(gl, vertices, 3, "position", shader_program);
    utils::set_float_buffer_attribute(gl, normals, 3, "normal", shader_program);
    utils::set_float_buffer_attribute(gl, tex_coords, 2, "texcoord", shader_program);
}

pub fn setup_vertices(
    gl: &WebGl2RenderingContext,
    vertices: &[f32],
    shader_program: &WebGlProgram,
) {
    utils::set_float_buffer_attribute(gl, vertices, 3, "position", shader_program);
}
#[wasm_bindgen]
pub struct GraphicsContext {
    gl: WebGl2RenderingContext,
    program: WebGlProgram,
    state: BoidUniverse,
    bodies: Vec<RenderObject>,
    materials: Vec<MaterialUniforms>,
    projection: nalgebra::Perspective3<f32>,
    look_at: Vector3<f32>,
    height: f32,
    width: f32,
    zoom: f32,
    yaw: f32,
    pitch: f32,
    turn_speed: f32,
    camera_position: Vector3<f32>,
    last_time_stamp: f32,
    max_time_step: f32,
    time_step_accumulator: f32,
    uniforms: GlobalUniforms,
}

pub trait RenderContext {
    fn raw_tick(&mut self, ms: f32);
    fn set_camera_position(&mut self, x: f32, y: f32, z: f32);

    fn set_cohesion(&mut self, c: f32);
    fn set_matching(&mut self, m: f32);
    fn set_avoid(&mut self, a: f32);
}

#[wasm_bindgen]
impl GraphicsContext {
    pub fn new(canvas_id: &str, height: f32, width: f32) -> Self {
        utils::set_panic_hook();
        let gl = init_webgl_context(canvas_id).unwrap();
        //let skybox_program = setup_skybox_shaders(&gl).unwrap();
        let program = setup_shaders(&gl).unwrap();
        //let mut uniform_locs = HashMap::new();

        for u in [
            "u_worldViewProjection",
            "u_world",
            "u_viewInverse",
            "u_worldInverseTranspose",
            "u_lightWorldPos",
            "u_diffuse",
            "u_specular",
            "u_shininess",
            "u_colorMult",
        ] {
            let res = gl.get_uniform_location(&program, u);

            if res.is_none() {
                log!("error getting uniform loc: {}", u);
            } else {
                log!("success getting uniform loc: {}", u);
            }
            //let loc = res.unwrap();
            //uniform_locs.insert(u, loc);
        }

        let shape = RenderShape::Cube;

        let projection =
            nalgebra::Perspective3::new(width / height, std::f32::consts::FRAC_PI_2, 0.001, 2000.0);
        let sprite_sheet: HtmlImageElement = utils::get_sprite_sheet("spritesheet").unwrap();
        let texture: WebGlTexture =
            utils::load_image_into_3_d_texture(&gl, &sprite_sheet, 13, 7).unwrap();

        let tex_coordinate_bounds = utils::get_sprite_coordinates(
            sprite_sheet.natural_width(),
            sprite_sheet.natural_height(),
            0,
            0,
            15,
            7,
        );
        load_attributes(&gl, &program, shape);

        let bodies = generate_bodies(1, shape);
        let global_uniforms = GlobalUniforms {
            world_view_projection: Matrix4::identity(),
            view_projection: Matrix4::identity(),
            view_inverse: Matrix4::identity(),
            light_world_pos: Vector3::new(-50.0, 30.0, 100.0),
            light_color: nalgebra::Vector4::new(1.0, 1.0, 1.0, 1.0),
            sprite_sheet,
            texture,
        };
        let materials = generate_materials(&gl, &global_uniforms, &bodies);
        GraphicsContext {
            gl: gl.clone(),
            state: BoidUniverse::new(),
            bodies,
            materials,
            program,
            //skybox_program,
            uniforms: global_uniforms,
            turn_speed: 0.1,
            projection,
            camera_position: Vector3::new(0.0, 0.0, 5.0),
            look_at: Vector3::new(0.0, 0.0, 0.0),
            height,
            width,
            pitch: 0.0,
            yaw: 0.0,
            zoom: 0.0,
            last_time_stamp: 0.0,
            max_time_step: 1.0 / 60.0,
            time_step_accumulator: 0.0,
        }
    }
    pub fn calculate_shared_uniforms(&mut self) {
        self.uniforms.view_inverse = Matrix4::look_at_rh(
            &Point3::from(self.camera_position),
            &Point3::from(self.look_at),
            &Vector3::y_axis(),
        );
        self.uniforms.view_projection = self.projection.as_matrix() * self.uniforms.view_inverse;
    }

    pub fn set_max_timestep(&mut self, ms: f32) {
        self.max_time_step = ms;
    }

    pub fn get_max_timestep(&self) -> f32 {
        self.max_time_step
    }
    pub fn raw_tick(&mut self, ms: f32) {
        let t = ms * 0.001;
        let mut frame_time = t - self.last_time_stamp;
        if frame_time > self.max_time_step {
            frame_time = self.max_time_step;
        }
        self.last_time_stamp = t;
        let mut last_state = self.bodies.clone();
        self.time_step_accumulator += frame_time;
        while self.time_step_accumulator >= self.max_time_step {
            last_state = update_bodies(last_state.clone(), self.max_time_step);
            self.last_time_stamp += self.max_time_step;
            self.time_step_accumulator -= self.max_time_step;
        }
        let alpha = self.time_step_accumulator / self.max_time_step;

        for (i, b) in self.bodies.iter_mut().enumerate() {
            b.orientation = (last_state[i].orientation).slerp(&b.orientation, alpha);
            b.world_position = (last_state[i].world_position).lerp(&b.world_position, alpha);
        }
        self.calculate_shared_uniforms();
        self.draw();
    }

    pub fn rotate_camera(&mut self, dx: f32, dy: f32) {
        self.pitch = -(dy * self.turn_speed).to_radians();
        self.pitch = self.pitch.clamp(-f32::consts::PI, f32::consts::PI);
        self.yaw = -(dx * self.turn_speed).to_radians();

        let y_quat = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), self.yaw);
        let p_quat = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), self.pitch);
        //self.direction = (y_quat * (p_quat * Vector3::from_row_slice(&self.direction))).into();
    }

    pub fn set_canvas_dimensions(&mut self, height: f32, width: f32) {
        let _ = &self.gl.viewport(0, 0, height as i32, width as i32);
        self.height = height;
        self.width = width;
    }

    pub fn zoom_camera(&mut self, delta: f32) {
        self.zoom += delta;
    }

    pub fn draw(&mut self) {
        self.draw_init();
        for i in 0..self.bodies.len() {
            draw_object(
                &self.gl,
                &self.program,
                &self.bodies[i],
                &self.materials[i],
                &self.uniforms,
            )
        }
    }

    pub fn set_camera_position(&mut self, x: f32, y: f32, z: f32) {
        self.camera_position = Vector3::new(x, y, z);
    }

    pub fn set_cohesion(&mut self, c: f32) {
        self.state.set_cohesion(c);
    }

    pub fn set_matching(&mut self, m: f32) {
        self.state.set_matching(m);
    }

    pub fn set_avoid(&mut self, a: f32) {
        self.state.set_avoid(a);
    }

    pub fn move_camera(&mut self, right: f32, up: f32, forward: f32) {
        let forward = (self.look_at - self.camera_position)
            .normalize()
            .scale(forward);
        let net = forward + forward.cross(&Vector3::y_axis()).scale(right);
        self.camera_position += net;
    }

    pub fn draw_init(&mut self) {
        self.gl.clear_color(0.0, 0.0, 0.0, 0.0);
        self.gl.clear_depth(1.0);
        self.gl.enable(WebGl2RenderingContext::DEPTH_TEST);
        self.gl.depth_func(WebGl2RenderingContext::LEQUAL);
        self.gl.clear(
            WebGl2RenderingContext::COLOR_BUFFER_BIT | WebGl2RenderingContext::DEPTH_BUFFER_BIT,
        );
    }
}
pub fn generate_materials(
    gl: &WebGl2RenderingContext,
    global_uniforms: &GlobalUniforms,
    bodies: &Vec<RenderObject>,
) -> Vec<MaterialUniforms> {
    let rand = rand::thread_rng();

    let color = ecolor::rgb_from_hsv(((rand::random::<f32>() * 240.0) as f32, 0.5, 1.0));
    bodies
        .iter()
        .map(|_body: &RenderObject| MaterialUniforms {
            diffuse_texture_unit: 0,
            color_mult: nalgebra::Vector4::new(color[0], color[1], color[2], 1.0),
            shininess: 250.0,
            specular_factor: 0.5,
            specular: nalgebra::Vector4::new(1.0, 1.0, 1.0, 1.0),
            face_index: [0, 1, 2, 3, 4, 5],
        })
        .collect()
}
pub fn generate_bodies(count: usize, shape: RenderShape) -> Vec<RenderObject> {
    let mut rand = rand::thread_rng();

    log!("Generating bodies: {}", count);
    (0..count)
        .map(|_| {
            let rand_pos: Vector3<f32> = Vector3::new(0.0, 0.0, 0.0);
            RenderObject {
                shape,
                world_position: rand_pos,
                orientation: UnitQuaternion::identity(),
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderShape {
    Triangle,
    Cube,
    Sphere,
    Quad,
    Tetrahedron,
}

#[wasm_bindgen]
#[derive(Debug, Clone, PartialEq)]
pub struct RenderObject {
    shape: RenderShape,
    world_position: Vector3<f32>,
    orientation: UnitQuaternion<f32>,
}

pub struct GlobalUniforms {
    world_view_projection: Matrix4<f32>,
    view_projection: Matrix4<f32>,
    view_inverse: Matrix4<f32>,
    light_world_pos: Vector3<f32>,
    light_color: nalgebra::Vector4<f32>,
    sprite_sheet: HtmlImageElement,
    texture: WebGlTexture,
}

pub struct MaterialUniforms {
    color_mult: nalgebra::Vector4<f32>,
    specular: nalgebra::Vector4<f32>,
    shininess: f32,
    specular_factor: f32,
    diffuse_texture_unit: u32,
    face_index: [u32; 6],
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
            setup_vertices_indices(
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
            let face_indices: [u32; 28] = [
                0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
            ];
            let indices: [u16; 36] = [
                0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15,
                16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
            ];
            setup_attributes_2d_array(
                gl,
                &vertices,
                &tex_coord,
                &normals,
                &indices,
                &face_indices,
                shader_program,
            );
        }
        RenderShape::Sphere => {}
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

            setup_vertices_indices(
                gl,
                &vertices,
                &tex_coord,
                &normals,
                &indices,
                shader_program,
            );
        }
        RenderShape::Tetrahedron => {}
    }
}

pub fn draw_object(
    gl: &WebGl2RenderingContext,
    shader_program: &WebGlProgram,
    obj: &RenderObject,
    materials: &MaterialUniforms,
    globals: &GlobalUniforms,
) {
    //let world = Matrix4::new_translation(&obj.world_position);
    let world = Matrix4::from(obj.orientation.to_homogeneous());
    let world_view_projection = globals.view_projection * world;
    let world_inverse_transpose = world.try_inverse().unwrap().transpose();
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
        match obj.shape {
            RenderShape::Cube => 72_i32,
            RenderShape::Sphere => 12_i32,
            RenderShape::Tetrahedron => 12_i32,
            RenderShape::Quad => 6_i32,
            RenderShape::Triangle => 3_i32,
        },
        WebGl2RenderingContext::UNSIGNED_SHORT,
        0,
    );
}
