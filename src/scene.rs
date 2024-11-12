use crate::boids;
use crate::RenderShape;
use core::f32;
use nalgebra::UnitQuaternion;
use nalgebra::Vector3;
use rand::thread_rng;
use rand::Rng;
use slotmap::{SecondaryMap, SlotMap};
use std::collections::HashSet;

#[derive(Clone, PartialEq)]
pub enum BasicUpdate {
    Rotate(f32),
    Bounce(f32),
    Scale(f32),
    Wiggle(f32),
    PhysicsBody,
    Boid,
    None,
}
pub struct SceneNode {
    target_pos: Vector3<f32>,
    target_scale: Vector3<f32>,
    target_rot: UnitQuaternion<f32>,
    velocity: Vector3<f32>,
    acceleration: Vector3<f32>,
}
impl SceneNode {
    pub fn new(pos: Vector3<f32>, scale: Vector3<f32>, rot: UnitQuaternion<f32>) -> Self {
        SceneNode {
            target_pos: pos,
            target_scale: scale,
            target_rot: rot,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            acceleration: Vector3::new(0.0, 0.0, 0.0),
        }
    }

    pub fn update_forces(&mut self, delta: f32) {
        self.velocity += self.acceleration * delta;
        self.target_pos += self.velocity * delta;
        self.target_rot =
            nalgebra::UnitQuaternion::face_towards(&self.velocity, &Vector3::y_axis());
    }
}

#[derive(Clone)]
pub struct Node {
    scale: nalgebra::Vector3<f32>,
    translation: nalgebra::Vector3<f32>,
    rotation: nalgebra::UnitQuaternion<f32>,
    pub global_transform: nalgebra::Matrix4<f32>,
    children: Vec<NodeID>,
    update: BasicUpdate,
}

#[derive(Clone)]
pub struct RenderNode {
    pub color_mult: nalgebra::Vector4<f32>,
    pub specular: nalgebra::Vector4<f32>,
    pub shininess: f32,
    pub specular_factor: f32,
    pub diffuse_texture_unit: i32,
    pub face_index: [u32; 6],
    pub shape: RenderShape,
}
impl RenderNode {
    pub fn new_cube(face: u32) -> Self {
        let mut rng = thread_rng();
        let base_color = rng.gen_range(0.0..240.0);
        let color =
            ecolor::rgb_from_hsv((rng.gen_range(base_color..(base_color + 120.0)), 0.5, 1.0));
        RenderNode {
            shape: RenderShape::Cube,
            diffuse_texture_unit: 0,
            color_mult: nalgebra::Vector4::new(color[0], color[1], color[2], 1.0),
            shininess: rng.gen_range(0.0..500.0),
            specular_factor: rng.gen_range(0.0..1.0),
            specular: nalgebra::Vector4::new(1.0, 1.0, 1.0, 1.0),
            face_index: [face, face, face, face, face, face],
        }
    }
    pub fn new_random() -> Self {
        let mut rng = thread_rng();
        let base_color = rng.gen_range(0.0..240.0);
        let color =
            ecolor::rgb_from_hsv((rng.gen_range(base_color..(base_color + 120.0)), 0.5, 1.0));
        let face = rng.gen_range(0..91);
        RenderNode {
            shape: RenderShape::Cube,
            diffuse_texture_unit: 0,
            color_mult: nalgebra::Vector4::new(color[0], color[1], color[2], 1.0),
            shininess: rng.gen_range(0.0..500.0),
            specular_factor: rng.gen_range(0.0..1.0),
            specular: nalgebra::Vector4::new(1.0, 1.0, 1.0, 1.0),
            face_index: [face, face, face, face, face, face],
        }
    }
}

pub type NodeID = slotmap::DefaultKey;
impl Default for Node {
    fn default() -> Self {
        Node::new()
    }
}
impl Node {
    pub fn new() -> Self {
        Node {
            update: BasicUpdate::None,
            scale: nalgebra::Vector3::new(1.0, 1.0, 1.0),
            translation: nalgebra::Vector3::new(0.0, 0.0, 0.0),
            rotation: nalgebra::UnitQuaternion::identity(),
            global_transform: nalgebra::Matrix4::identity(),
            children: vec![],
        }
    }

    pub fn new_random(world_size: Vector3<f32>, update_type: BasicUpdate) -> Node {
        let mut rng = thread_rng();
        let translation: nalgebra::Vector3<f32> = nalgebra::Vector3::new(
            rng.gen_range(-world_size.x * 0.5..world_size.x * 0.5),
            rng.gen_range(-world_size.y * 0.5..world_size.y * 0.5),
            rng.gen_range(-world_size.z * 0.5..world_size.z * 0.5),
        );
        let scale: nalgebra::Vector3<f32> = nalgebra::Vector3::new(
            rng.gen_range(0.5..2.0),
            rng.gen_range(0.5..2.0),
            rng.gen_range(0.5..2.0),
        );
        let rotation = nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), 0.0);
        Node {
            update: update_type,
            scale,
            translation,
            rotation,
            global_transform: nalgebra::Matrix4::identity(),
            children: vec![],
        }
    }
    pub fn new_with_start(pos: Vector3<f32>, rot: nalgebra::UnitQuaternion<f32>) -> Self {
        Node {
            update: BasicUpdate::None,
            scale: Vector3::new(1.0, 1.0, 1.0),
            translation: pos,
            rotation: rot,
            global_transform: nalgebra::Matrix4::identity(),
            children: vec![],
        }
    }

    pub fn update(&mut self, delta: f32, scene_info: &SceneNode) {
        //self.translation += (scene_info.target_pos - self.translation).scale(delta);
        self.translation = self.translation.lerp(&scene_info.target_pos.xyz(), delta);
        self.rotation = self.rotation.slerp(&scene_info.target_rot, delta);
        self.scale = scene_info.target_scale;
    }
    pub fn update_with_forces(
        &mut self,
        delta: f32,
        target_position: &Vector3<f32>,
        target_rotation: &UnitQuaternion<f32>,
    ) {
        self.translation = self.translation.lerp(target_position, delta);
        self.rotation = self.rotation.slerp(target_rotation, delta);
    }

    pub fn translate_update(&mut self, delta: f32) {
        self.translation += Vector3::identity().scale((std::f32::consts::FRAC_PI_8 * delta).sin())
    }

    pub fn calculate_global_transform(&mut self, parent_transform: &nalgebra::Matrix4<f32>) {
        let local_transformation = nalgebra::Isometry3::from_parts(
            nalgebra::Translation::from(self.translation),
            self.rotation,
        )
        .to_homogeneous()
        .append_nonuniform_scaling(&self.scale);
        self.global_transform = parent_transform * local_transformation;
    }
}

pub struct World {
    pub root_id: NodeID,
    pub camera_id: NodeID,
    pub size: nalgebra::Vector3<f32>,
    arena: slotmap::SlotMap<NodeID, Node>,
    materials: slotmap::SecondaryMap<NodeID, RenderNode>,
    scene_info: slotmap::SecondaryMap<NodeID, SceneNode>,
    pub camera_info: CameraParameters,
    pub bonus_parameters: boids::BoidsParameters,
}
pub struct CameraParameters {
    pub target_id: NodeID,
    pub gimbal_y: NodeID,
    pub perpective: nalgebra::Perspective3<f32>,
}

impl World {
    pub fn get_node_count(&self) -> usize {
        self.arena.len()
    }
    // run at the beginning of loop
    pub fn update(&mut self, delta: f32) {
        let mut stack: Vec<NodeID> = vec![self.root_id];

        let mut purge_keys: HashSet<NodeID> = HashSet::new();
        while let Some(current_id) = stack.pop() {
            {
                let current_node = self.arena[current_id].clone();
                for i in 0..current_node.children.len() {
                    let child_node = match self.arena.get_mut(current_node.children[i]) {
                        Some(node) => node,
                        None => {
                            purge_keys.insert(current_node.children[i]);
                            continue;
                        }
                    };
                    if let Some(scene_info) = self.scene_info.get_mut(current_node.children[i]) {
                        match child_node.update {
                            BasicUpdate::Boid | BasicUpdate::PhysicsBody => {
                                scene_info.update_forces(delta);
                            }
                            _ => {}
                        }

                        child_node.update(delta, scene_info);
                    }
                }
            }
            let s = self.arena.get_mut(current_id).unwrap();
            let _ = s
                .children
                .iter_mut()
                .filter(|node_id| purge_keys.contains(node_id));
            stack.extend(&s.children);
        }
    }
    pub fn generate_nested_bodies(&mut self, mut depth: usize, face: u32) {
        let mut parent = self.root_id;
        while depth > 0 {
            depth -= 1;

            let r = RenderNode::new_cube(face);
            let new_node = self.new_random_node();
            let id = self.add_nested_body(parent, new_node);
            self.materials.insert(id, r.clone());
            parent = id;
        }
    }

    pub fn generate_boids(&mut self, count: usize, face: u32) {
        let r = RenderNode::new_cube(face);
        (0..count).for_each(|_| {
            let new_node = Node::new_random(self.size, BasicUpdate::Boid);
            let id = self.add_body(new_node.clone());

            self.scene_info.insert(
                id,
                SceneNode::new(new_node.translation, new_node.scale, new_node.rotation),
            );
            self.materials.insert(id, r.clone());
        });
    }

    pub fn generate_bodies(&mut self, count: usize, face: u32) {
        let r = RenderNode::new_cube(face);
        (0..count).for_each(|_| {
            let new_node = Node::new_random(self.size, BasicUpdate::PhysicsBody);
            let id = self.add_body(new_node.clone());

            self.scene_info.insert(
                id,
                SceneNode::new(new_node.translation, new_node.scale, new_node.rotation),
            );
            self.materials.insert(id, r.clone());
        });
    }
    // run right before the draw calls
    pub fn calculate_final_render_objects(&mut self) -> Vec<NodeID> {
        let mut seen: Vec<NodeID> = vec![];
        let mut stack: Vec<NodeID> = vec![self.root_id];

        let mut purge_keys: HashSet<NodeID> = HashSet::new();
        while let Some(current_id) = stack.pop() {
            {
                let current_node = self.arena[current_id].clone();
                let parent_transform = &current_node.global_transform;
                for i in 0..current_node.children.len() {
                    let child_node = match self.arena.get_mut(current_node.children[i]) {
                        Some(node) => node,
                        None => {
                            purge_keys.insert(current_node.children[i]);
                            continue;
                        }
                    };
                    child_node.calculate_global_transform(parent_transform);
                    if self.materials.contains_key(current_node.children[i]) {
                        seen.push(current_node.children[i]);
                    }
                }
            }
            let s = self.arena.get_mut(current_id).unwrap();
            let _ = s
                .children
                .iter_mut()
                .filter(|node_id| purge_keys.contains(node_id));
            stack.extend(&s.children);
        }
        seen
    }

    pub fn get_node(&self, id: NodeID) -> &Node {
        &self.arena[id]
    }
    pub fn get_materials_node(&self, id: NodeID) -> &RenderNode {
        &self.materials[id]
    }
    pub fn new_random_node(&mut self) -> Node {
        let mut rng = thread_rng();
        let translation: nalgebra::Vector3<f32> = nalgebra::Vector3::new(
            rng.gen_range(-self.size.x * 0.5..self.size.x * 0.5),
            rng.gen_range(-self.size.y * 0.5..self.size.y * 0.5),
            rng.gen_range(-self.size.z * 0.5..self.size.z * 0.5),
        );
        let scale: nalgebra::Vector3<f32> = nalgebra::Vector3::new(
            rng.gen_range(0.5..2.0),
            rng.gen_range(0.5..2.0),
            rng.gen_range(0.5..2.0),
        );
        let rotation = nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), 0.0);
        Node {
            update: BasicUpdate::Rotate(1.0),
            scale,
            translation,
            rotation,
            global_transform: nalgebra::Matrix4::identity(),
            children: vec![],
        }
    }
    pub fn add_node_at(&mut self, pos: Vector3<f32>) {
        let mut node = Node::new();

        node.translation = pos;
        let id = self.add_body(node);
        self.materials.insert(id, RenderNode::new_cube(3));
    }
    pub fn add_node_at_view(&mut self) {
        let mut node = Node::new();

        let target_pos = nalgebra::Vector4::from(
            self.arena
                .get(self.camera_info.target_id)
                .unwrap()
                .global_transform
                .index((.., 3)),
        );
        node.translation = target_pos.xyz();
        let id = self.add_body(node);
        self.materials.insert(id, RenderNode::new_cube(3));
    }

    pub fn new() -> Self {
        let mut arena = SlotMap::new();
        let size = Vector3::new(200.0, 200.0, 200.0);
        let root_id = arena.insert(Node::new());
        let camera_id = arena.insert(Node::new_with_start(
            Vector3::new(0.0, 0.0, 0.0),
            UnitQuaternion::identity(),
        ));

        let target_id = arena.insert(Node::new_with_start(
            Vector3::new(0.0, 0.0, -1.0),
            UnitQuaternion::identity(),
        ));
        let gimbal_y = arena.insert(Node::new_with_start(
            Vector3::new(0.0, 0.0, 0.0),
            UnitQuaternion::identity(),
        ));
        arena[root_id].children.push(camera_id);
        arena[camera_id].children.push(gimbal_y);
        arena[gimbal_y].children.push(target_id);
        let bonus_parameters = boids::BoidsParameters::new();
        World {
            root_id,
            camera_id,
            arena,
            materials: SecondaryMap::new(),
            scene_info: SecondaryMap::new(),
            size,
            bonus_parameters,
            camera_info: CameraParameters {
                target_id,
                gimbal_y,
                perpective: nalgebra::Perspective3::new(
                    16.0 / 9.0,
                    std::f32::consts::FRAC_PI_3,
                    0.001,
                    2000.0,
                ),
            },
        }
    }
    pub fn add_body(&mut self, node: Node) -> NodeID {
        let id = self.arena.insert(node);
        self.arena[self.root_id].children.push(id);
        id
    }
    pub fn add_nested_body(&mut self, parent_id: NodeID, node: Node) -> NodeID {
        let id = self.arena.insert(node);
        self.arena[parent_id].children.push(id);
        id
    }

    pub fn get_camera_view_inverse(&self) -> nalgebra::Matrix4<f32> {
        let camera_node = self.arena.get(self.camera_id).unwrap();

        let camera_world_pos: nalgebra::Vector4<f32> =
            nalgebra::Vector4::from(camera_node.global_transform.index((.., 3)));
        let target_pos = nalgebra::Vector4::from(
            self.arena
                .get(self.camera_info.target_id)
                .unwrap()
                .global_transform
                .index((.., 3)),
        );
        nalgebra::Matrix4::look_at_rh(
            &nalgebra::Point3::from(camera_world_pos.xyz()),
            &nalgebra::Point3::from(target_pos.xyz()),
            &nalgebra::Vector3::y_axis(),
        )
    }
    pub fn move_camera(&mut self, displacement: Vector3<f32>) {
        let camera_node = self.arena.get(self.camera_id).unwrap();
        let look_at = self.arena.get(self.camera_info.target_id).unwrap();

        let camera_world_pos: nalgebra::Vector4<f32> =
            nalgebra::Vector4::from(camera_node.global_transform.index((.., 3)));
        let target_pos = nalgebra::Vector4::from(look_at.global_transform.index((.., 3)));

        let forward = (target_pos - camera_world_pos).xyz().normalize();

        let net =
            forward.scale(displacement.z) + forward.cross(&Vector3::y_axis().scale(displacement.x));
        self.arena[self.camera_id].translation += net;
    }

    pub fn rotate_camera(&mut self, dx: f32, dy: f32) {
        self.arena[self.camera_id].rotation *= UnitQuaternion::from_axis_angle(
            &Vector3::y_axis(),
            f32::consts::FRAC_PI_8 * dx * -0.001,
        );
        self.arena[self.camera_info.gimbal_y].rotation *= UnitQuaternion::from_axis_angle(
            &Vector3::x_axis(),
            f32::consts::FRAC_PI_8 * dy * -0.001,
        );
    }

    pub fn zoom_camera(&mut self, delta: f32) {
        self.camera_info
            .perpective
            .set_fovy(self.camera_info.perpective.fovy() + delta);
    }

    pub fn update_aspect_ratio(&mut self, width: f32, height: f32) {
        self.camera_info.perpective.set_aspect(width / height);
    }
    pub fn get_all_boids(&self) -> Vec<NodeID> {
        let mut seen: Vec<NodeID> = vec![];
        let mut stack: Vec<NodeID> = vec![self.root_id];

        while let Some(current_id) = stack.pop() {
            let current_node = self.arena[current_id].clone();
            if current_node.update == BasicUpdate::Boid {
                seen.push(current_id);
            }
            stack.extend(current_node.children);
        }
        seen
    }

    pub fn boids_update(&mut self, delta: f32) {
        let boid_ids = self.get_all_boids();
        let next_ids = boid_ids.clone();
        for b in boid_ids {
            let mut avoid_v = Vector3::zeros();
            let mut avg_v = Vector3::zeros();
            let mut avg_p = Vector3::zeros();
            let mut count = 0;
            let current_pos = self.scene_info.get(b).unwrap().target_pos;
            for boid_id in &next_ids {
                if *boid_id == b {
                    continue;
                }
                let neighbor = self.scene_info.get(*boid_id).unwrap();
                let dist = &current_pos.metric_distance(&neighbor.target_pos);
                if dist < &self.bonus_parameters.protected_range {
                    avoid_v += current_pos - neighbor.target_pos;
                }
                if dist < &self.bonus_parameters.visual_range {
                    avg_v += neighbor.velocity;
                    avg_p += neighbor.target_pos;
                    count += 1;
                }
            }
            let current_b = self.scene_info.get_mut(b).unwrap();
            avg_v = avg_v.scale(1.0 / (count as f32));
            avg_p = avg_p.scale(1.0 / (count as f32));
            current_b.velocity += avoid_v.scale(self.bonus_parameters.avoid_factor)
                + (avg_v - current_b.velocity).scale(self.bonus_parameters.matching_factor)
                + (avg_p - current_b.target_pos).scale(self.bonus_parameters.cohesion_factor);
            if current_b.target_pos.x < -self.size.x / 2.0 {
                current_b.velocity.x += self.bonus_parameters.turn_factor;
            }
            if current_b.target_pos.y < -self.size.y / 2.0 {
                current_b.velocity.y += self.bonus_parameters.turn_factor;
            }

            if current_b.target_pos.z < -self.size.z / 2.0 {
                current_b.velocity.z += self.bonus_parameters.turn_factor;
            }
            if current_b.target_pos.x > self.size.x / 2.0 {
                current_b.velocity.x -= self.bonus_parameters.turn_factor;
            }
            if current_b.target_pos.y > self.size.y / 2.0 {
                current_b.velocity.y -= self.bonus_parameters.turn_factor;
            }

            if current_b.target_pos.z > self.size.z / 2.0 {
                current_b.velocity.z -= self.bonus_parameters.turn_factor;
            }
            //if i % 2 == 0 {
            //    current_b.velocity.x = current_b.velocity.x * (1.0 - self.bonus_parameters.bias)
            //        + (1.0 * self.bonus_parameters.bias);
            //} else {
            //    current_b.velocity.x = current_b.velocity.x * (1.0 - self.bonus_parameters.bias)
            //        + (-1.0 * self.bonus_parameters.bias);
            //}

            if current_b.velocity.magnitude() < self.bonus_parameters.min_speed {
                current_b
                    .velocity
                    .set_magnitude(self.bonus_parameters.min_speed);
            } else {
                current_b.velocity = current_b
                    .velocity
                    .cap_magnitude(self.bonus_parameters.max_speed);
            }
        }
        self.update(delta);
    }
}
