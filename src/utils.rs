use crate::RenderObject;
use nalgebra::{dimension, Matrix4, Unit, UnitQuaternion, Vector3};
use std::f32;
use wasm_bindgen::prelude::*;
use web_sys::{HtmlImageElement, WebGl2RenderingContext, WebGlProgram, WebGlTexture};
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
    texture_unit: u32,
    texture: &WebGlTexture,
    uniform_name: &str,
) {
    let res = gl.get_uniform_location(shader_program, uniform_name);
    if res.is_none() {
        //log!("uniform not found :{}", uniform_name);
        return;
    }
    gl.active_texture(WebGl2RenderingContext::TEXTURE0 + texture_unit);
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D_ARRAY, Some(texture));
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
    let max_depth = gl
        .get_parameter(WebGl2RenderingContext::MAX_ARRAY_TEXTURE_LAYERS)
        .unwrap();
    log!("max_depth: {:?}", max_depth);
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
            log!("({}, {})", x_offset, y_offset);
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

            gl.tex_parameteri(
                WebGl2RenderingContext::TEXTURE_2D_ARRAY,
                WebGl2RenderingContext::TEXTURE_MIN_FILTER,
                WebGl2RenderingContext::LINEAR as i32,
            );
            gl.tex_parameteri(
                WebGl2RenderingContext::TEXTURE_2D_ARRAY,
                WebGl2RenderingContext::TEXTURE_MAG_FILTER,
                WebGl2RenderingContext::LINEAR as i32,
            );
            gl.tex_parameteri(
                WebGl2RenderingContext::TEXTURE_2D_ARRAY,
                WebGl2RenderingContext::TEXTURE_WRAP_S,
                WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameteri(
                WebGl2RenderingContext::TEXTURE_2D_ARRAY,
                WebGl2RenderingContext::TEXTURE_WRAP_T,
                WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
            );
        }
    }
    log!("mewo");

    Ok(texture)
}
pub fn load_image_into_texture(
    gl: &WebGl2RenderingContext,
    image: &HtmlImageElement,
) -> Result<WebGlTexture, JsValue> {
    let texture = create_texture(gl);
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));
    gl.tex_image_2d_with_u32_and_u32_and_html_image_element(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        WebGl2RenderingContext::RGBA.try_into().unwrap(),
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        &image,
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

pub fn recompile() {}
pub fn set_float_buffer_attribute(
    gl: &WebGl2RenderingContext,
    data: &[f32],
    size: i32,
    name: &str,
    shader_program: &WebGlProgram,
) {
    let buffer = gl.create_buffer().unwrap();
    let array = unsafe { js_sys::Float32Array::view(data) };
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&buffer));
    gl.buffer_data_with_array_buffer_view(
        WebGl2RenderingContext::ARRAY_BUFFER,
        &array,
        WebGl2RenderingContext::STATIC_DRAW,
    );
    let loc = gl.get_attrib_location(shader_program, name);
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&buffer));

    gl.enable_vertex_attrib_array(loc as u32);
    gl.vertex_attrib_pointer_with_i32(loc as u32, size, WebGl2RenderingContext::FLOAT, false, 0, 0);
}

pub fn set_int_buffer_attribute(
    gl: &WebGl2RenderingContext,
    data: &[u32],
    size: i32,
    name: &str,
    shader_program: &WebGlProgram,
) {
    let buffer = gl.create_buffer().unwrap();
    let array = unsafe { js_sys::Uint32Array::view(data) };
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&buffer));
    gl.buffer_data_with_array_buffer_view(
        WebGl2RenderingContext::ARRAY_BUFFER,
        &array,
        WebGl2RenderingContext::STATIC_DRAW,
    );
    let loc = gl.get_attrib_location(shader_program, name);
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&buffer));
    gl.enable_vertex_attrib_array(loc as u32);
    gl.vertex_attrib_pointer_with_i32(
        loc as u32,
        size,
        WebGl2RenderingContext::UNSIGNED_INT,
        false,
        0,
        0,
    );
}

pub fn update_bodies(mut bodies: Vec<RenderObject>, delta: f32) -> Vec<RenderObject> {
    for (i, b) in bodies.iter_mut().enumerate() {
        let target_orientation = b.orientation
            * UnitQuaternion::from_axis_angle(
                &Unit::new_normalize(Vector3::new(1.0, 1.0, 1.0)),
                f32::consts::FRAC_PI_2,
            );
        b.orientation = b.orientation.slerp(&target_orientation, delta);
    }
    bodies
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
