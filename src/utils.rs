use crate::RenderObject;
use nalgebra::{Matrix4, Unit, UnitQuaternion, Vector3};
use std::f32;
use wasm_bindgen::prelude::*;
use web_sys::{HtmlImageElement, WebGl2RenderingContext, WebGlProgram, WebGlShader, WebGlTexture};
extern crate web_sys;

#[macro_export]
macro_rules! log {
    ($( $t:tt)*) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}
pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

pub fn new_webgl_context(canvas_id: &str) -> Result<WebGl2RenderingContext, JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id(canvas_id).unwrap();

    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;

    let gl: WebGl2RenderingContext = canvas
        .get_context("webgl2")?
        .unwrap()
        .dyn_into::<WebGl2RenderingContext>()
        .unwrap();
    gl.viewport(0, 0, canvas.width() as i32, canvas.height() as i32);
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

        in vec4 a_position;
        in vec3 a_normal;
        in vec2 a_texcoord;
        in uint a_faceId;

        out vec4 v_position;
        out vec2 v_texCoord;
        out vec3 v_normal;
        out vec3 v_surfaceToLight;
        out vec3 v_surfaceToView;
        flat out uint v_faceId;

        void main() {
            v_faceId = a_faceId;
            v_texCoord = a_texcoord;
            v_position = u_worldViewProjection * a_position;
            v_normal = (u_worldInverseTranspose * vec4(a_normal, 0.0)).xyz;
            v_surfaceToLight = u_lightWorldPos - (u_world * a_position).xyz;
            v_surfaceToView = (u_viewInverse[3] - (u_world * a_position)).xyz;
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
          color = vec4((u_lightColor * (diffuseColor * litR.y + u_specular * litR.z * u_specularFactor)).rgb, 1.0);
          vec4 ambient = 0.5 * u_lightColor;
          vec4 diffuse = max(dot(a_normal, surfaceToLight), 0.0) * u_lightColor;
          vec4 result = (diffuse + ambient) * diffuseColor;
          color = vec4(result.xyz, 1.0);
          //color = vec4(0.0, 0.0, 1.0, 1.0);
          //color = vec4(litR.yyy, 1);
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
    face_indices: &[u8],
    shader_program: &WebGlProgram,
) {
    let indices_buffer = gl.create_buffer().unwrap();
    gl.bind_buffer(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        Some(&indices_buffer),
    );
    let indices_array = unsafe { js_sys::Uint16Array::view(indices) };

    gl.buffer_data_with_array_buffer_view(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        &indices_array,
        WebGl2RenderingContext::STATIC_DRAW,
    );
    set_float_buffer_attribute(gl, vertices, 3, "a_position", shader_program);
    set_float_buffer_attribute(gl, normals, 3, "a_normal", shader_program);
    set_float_buffer_attribute(gl, tex_coords, 2, "a_texcoord", shader_program);
    set_int_buffer_attribute(gl, face_indices, 1, "a_faceId", shader_program);
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
    set_float_buffer_attribute(gl, vertices, 3, "a_position", shader_program);
    set_float_buffer_attribute(gl, normals, 3, "a_normal", shader_program);
    set_float_buffer_attribute(gl, tex_coords, 2, "a_texcoord", shader_program);
}

pub fn setup_vertices(
    gl: &WebGl2RenderingContext,
    vertices: &[f32],
    shader_program: &WebGlProgram,
) {
    set_float_buffer_attribute(gl, vertices, 3, "a_position", shader_program);
}
pub fn rotate_over_time(ms: f32) -> Matrix4<f32> {
    let axis_angle = Vector3::z() * f32::consts::FRAC_PI_2 * ms * 0.0005;

    Matrix4::from_scaled_axis(axis_angle)
}
pub fn write_uniform_uint_array(
    gl: &WebGl2RenderingContext,
    shader_program: &WebGlProgram,
    uniform_value: &[u32],
    uniform_name: &str,
) {
    let res = gl.get_uniform_location(shader_program, uniform_name);
    if res.is_none() {
        log!("uniform not found :{}", uniform_name);
        return;
    }
    let loc = res.unwrap();

    gl.uniform1uiv_with_u32_array(Some(&loc), uniform_value);
}
pub fn write_uniform_float(
    gl: &WebGl2RenderingContext,
    shader_program: &WebGlProgram,
    uniform_value: f32,
    uniform_name: &str,
) {
    let res = gl.get_uniform_location(shader_program, uniform_name);
    if res.is_none() {
        //log!("uniform not found :{}", uniform_name);
        return;
    }
    let loc = res.unwrap();
    gl.uniform1f(Some(&loc), uniform_value);
}
pub fn write_uniform_texture_unit(
    gl: &WebGl2RenderingContext,
    shader_program: &WebGlProgram,
    texture_unit: i32,
    texture: &WebGlTexture,
    uniform_name: &str,
) {
    let res = gl.get_uniform_location(shader_program, uniform_name);
    if res.is_none() {
        log!("uniform not found :{}", uniform_name);
        return;
    }
    let loc = res.unwrap();
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D_ARRAY, Some(texture));
    gl.uniform1i(Some(&loc), texture_unit);
}
pub fn write_uniform_vec4(
    gl: &WebGl2RenderingContext,
    shader_program: &WebGlProgram,
    uniform_value: &[f32],
    uniform_name: &str,
) {
    let res = gl.get_uniform_location(shader_program, uniform_name);
    if res.is_none() {
        //log!("uniform not found :{}", uniform_name);
        return;
    }
    let loc = res.unwrap();
    gl.uniform4fv_with_f32_array(Some(&loc), uniform_value);
}
pub fn write_uniform_vec3(
    gl: &WebGl2RenderingContext,
    shader_program: &WebGlProgram,
    uniform_value: &[f32],
    uniform_name: &str,
) {
    let res = gl.get_uniform_location(shader_program, uniform_name);
    if res.is_none() {
        //log!("uniform not found :{}", uniform_name);
        return;
    }
    let loc = res.unwrap();
    gl.uniform3fv_with_f32_array(Some(&loc), uniform_value);
}
pub fn write_uniform_mat4(
    gl: &WebGl2RenderingContext,
    shader_program: &WebGlProgram,
    uniform_value: &[f32],
    uniform_name: &str,
) {
    let res = gl.get_uniform_location(shader_program, uniform_name);
    if res.is_none() {
        //log!("uniform not found :{}", uniform_name);
        return;
    }
    let loc = res.unwrap();
    gl.uniform_matrix4fv_with_f32_array(Some(&loc), false, uniform_value);
}
pub fn get_sprite_sheet(image_id: &str) -> Result<HtmlImageElement, JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let image = document.get_element_by_id(image_id).unwrap();

    let image: HtmlImageElement = image.dyn_into::<HtmlImageElement>()?;
    Ok(image)
}

pub fn create_texture(gl: &WebGl2RenderingContext) -> WebGlTexture {
    let t = gl.create_texture().unwrap();
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&t));
    let color: &[u8] = &[0, 0, 255, 255];
    let _ = gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        WebGl2RenderingContext::RGBA.try_into().unwrap(),
        1,
        1,
        0,
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(color),
    );
    t
}
pub fn load_image_into_3_d_texture(
    gl: &WebGl2RenderingContext,
    image: &HtmlImageElement,
    x_count: u32,
    y_count: u32,
) -> Result<WebGlTexture, JsValue> {
    let texture = gl.create_texture().unwrap();
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D_ARRAY, Some(&texture));

    let sprite_size = 64;
    let depth: i32 = (x_count * y_count) as i32;
    gl.tex_storage_3d(
        WebGl2RenderingContext::TEXTURE_2D_ARRAY,
        1,
        WebGl2RenderingContext::RGBA8,
        sprite_size,
        sprite_size,
        depth,
    );
    gl.pixel_storei(
        WebGl2RenderingContext::UNPACK_ROW_LENGTH,
        image.natural_width().try_into().unwrap(),
    );
    for y in 0..y_count {
        for x in 0..x_count {
            let x_offset = x * sprite_size as u32;
            let y_offset = y * sprite_size as u32;
            let d = y * x_count + x;
            gl.pixel_storei(
                WebGl2RenderingContext::UNPACK_SKIP_PIXELS,
                x_offset.try_into().unwrap(),
            );

            gl.pixel_storei(
                WebGl2RenderingContext::UNPACK_SKIP_ROWS,
                y_offset.try_into().unwrap(),
            );
            //log!("({}, {})", x_offset, y_offset);
            gl.tex_sub_image_3d_with_html_image_element(
                WebGl2RenderingContext::TEXTURE_2D_ARRAY,
                0,
                0,
                0,
                d.try_into().unwrap(),
                sprite_size,
                sprite_size,
                1,
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                image,
            )
            .unwrap();
            //
            //gl.tex_parameteri(
            //    WebGl2RenderingContext::TEXTURE_2D_ARRAY,
            //    WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            //    WebGl2RenderingContext::LINEAR as i32,
            //);
            //gl.tex_parameteri(
            //    WebGl2RenderingContext::TEXTURE_2D_ARRAY,
            //    WebGl2RenderingContext::TEXTURE_MAG_FILTER,
            //    WebGl2RenderingContext::LINEAR as i32,
            //);
            //gl.tex_parameteri(
            //    WebGl2RenderingContext::TEXTURE_2D_ARRAY,
            //    WebGl2RenderingContext::TEXTURE_WRAP_S,
            //    WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
            //);
            //gl.tex_parameteri(
            //    WebGl2RenderingContext::TEXTURE_2D_ARRAY,
            //    WebGl2RenderingContext::TEXTURE_WRAP_T,
            //    WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
            //);
        }
    }

    Ok(texture)
}

pub fn load_image_into_texture(
    gl: &WebGl2RenderingContext,
    image: &HtmlImageElement,
) -> Result<WebGlTexture, JsValue> {
    let texture = create_texture(gl);
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));
    let _ = gl.tex_image_2d_with_u32_and_u32_and_html_image_element(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        WebGl2RenderingContext::RGBA.try_into().unwrap(),
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        image,
    );
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_MIN_FILTER,
        WebGl2RenderingContext::LINEAR.try_into().unwrap(),
    );
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_WRAP_S,
        WebGl2RenderingContext::CLAMP_TO_EDGE.try_into().unwrap(),
    );
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_WRAP_T,
        WebGl2RenderingContext::CLAMP_TO_EDGE.try_into().unwrap(),
    );
    gl.pixel_storei(WebGl2RenderingContext::UNPACK_FLIP_Y_WEBGL, 1);
    Ok(texture)
}

pub fn set_float_buffer_attribute(
    gl: &WebGl2RenderingContext,
    data: &[f32],
    size: i32,
    name: &str,
    shader_program: &WebGlProgram,
) {
    let loc = gl.get_attrib_location(shader_program, name);
    if loc < 0 {
        log!("ERROR: float attr_loc for {}:{}", name, loc);
        return;
    }

    let buffer = gl.create_buffer().unwrap();
    let array = unsafe { js_sys::Float32Array::view(data) };
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&buffer));
    gl.buffer_data_with_array_buffer_view(
        WebGl2RenderingContext::ARRAY_BUFFER,
        &array,
        WebGl2RenderingContext::STATIC_DRAW,
    );
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&buffer));

    gl.enable_vertex_attrib_array(loc as u32);
    gl.vertex_attrib_pointer_with_i32(loc as u32, size, WebGl2RenderingContext::FLOAT, false, 0, 0);
}

pub fn set_int_buffer_attribute(
    gl: &WebGl2RenderingContext,
    data: &[u8],
    size: i32,
    name: &str,
    shader_program: &WebGlProgram,
) {
    let buffer = gl.create_buffer().unwrap();
    let array = unsafe { js_sys::Uint8Array::view(data) };
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&buffer));
    gl.buffer_data_with_array_buffer_view(
        WebGl2RenderingContext::ARRAY_BUFFER,
        &array,
        WebGl2RenderingContext::STATIC_DRAW,
    );
    let loc = gl.get_attrib_location(shader_program, name);
    if loc < 0 {
        log!("ERROR: int attr_loc for {}:{}", name, loc);
    }
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&buffer));
    gl.enable_vertex_attrib_array(loc as u32);
    gl.vertex_attrib_i_pointer_with_i32(
        loc as u32,
        size,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        0,
        0,
    );
}

pub fn update_bodies(bodies: &mut [RenderObject], delta: f32) {
    for b in bodies.iter_mut() {
        let target_orientation = b.orientation
            * UnitQuaternion::from_axis_angle(
                &Unit::new_normalize(Vector3::new(1.0, 1.0, 1.0)),
                f32::consts::FRAC_PI_4,
            );
        b.orientation = b.orientation.slerp(&target_orientation, delta);
    }
}

pub fn get_sprite_coordinates(
    sheet_width: u32,
    sheet_height: u32,
    x: u32,
    y: u32,
    x_count: u32,
    y_count: u32,
) -> (f32, f32, f32, f32) {
    let sprite_width = (x_count as f32) / (sheet_width as f32);
    let sprite_height = (y_count as f32) / (sheet_height as f32);
    (
        x as f32 * sprite_width,
        y as f32 * sprite_height,
        (x as f32 + 1.0) * sprite_width,
        (y as f32 + 1.0) * sprite_height,
    )
}
