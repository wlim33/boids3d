use nalgebra::{Matrix4, Point3, Translation3, Vector3};
use rand::{thread_rng, Rng};
use wasm_bindgen::prelude::*;

use rand::distributions::Standard;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Boid {
    pos: Vector3<f32>,
    vel: Vector3<f32>,
    id: usize,
}

impl Boid {
    pub fn new(id: usize, x: f32, y: f32, z: f32) -> Boid {
        Boid {
            id,
            pos: Vector3::new(x, y, z),
            vel: Vector3::new(0.0, 0.0, 0.0),
        }
    }

    pub fn get_id(&self) -> usize {
        return self.id;
    }

    pub fn get_vel(&self) -> Vector3<f32> {
        return self.vel;
    }

    pub fn get_pos(&self) -> Vector3<f32> {
        return self.pos;
    }

    pub fn get_matrix(&self) -> Matrix4<f32> {
        let pos: Point3<f32> = Point3::from(self.pos);
        let look_at: Point3<f32> = Point3::from(pos + self.vel);
        Translation3::from(pos)
            .to_homogeneous()
            .try_inverse()
            .unwrap()
        //Matrix4::look_at_rh(&pos, &look_at, &Vector3::new(0.0, 1.0, 0.0))
        //    .try_inverse()
        //    .unwrap()
    }
}
#[wasm_bindgen]
pub struct BoidUniverse {
    matching: f32,
    avoid: f32,
    cohesion: f32,
    visual_range: f32,
    protected_range: f32,
    turn_factor: f32,
    max_speed: f32,
    min_speed: f32,
    bias: f32,
    max_bias: f32,
    bias_increment: f32,
    time_step_accumulator: f32,
    gravity: Vector3<f32>,
    boids: Boids,
    world_size: Vector3<f32>,
}

type Boids = Vec<Boid>;
impl Default for BoidUniverse {
    fn default() -> Self {
        BoidUniverse::new()
    }
}

#[wasm_bindgen]
impl BoidUniverse {
    pub fn new() -> Self {
        let mut rng = thread_rng();
        let boids_count = 15;
        let starting_positions: Vec<f32> = (&mut rng)
            .sample_iter(Standard)
            .take(boids_count * boids_count)
            .collect();

        let mut starting_boids: Vec<Boid> = vec![];
        for i in 0..boids_count {
            starting_boids.push(Boid::new(
                i,
                starting_positions[i],
                starting_positions[i + boids_count],
                starting_positions[i + boids_count * 2],
            ))
        }

        BoidUniverse {
            world_size: Vector3::new(72.0, 72.0, 72.0),
            matching: 0.2 * 10.0,
            cohesion: 0.0005 * 10.0,
            avoid: 0.05 * 10.0,
            visual_range: 40.0,
            protected_range: 8.0,
            turn_factor: 0.2,
            bias: 0.001,
            bias_increment: 0.00004,
            max_bias: 0.01,
            max_speed: 12.0,
            min_speed: 12.0,
            boids: starting_boids,
            time_step_accumulator: 0.0,
            gravity: Vector3::new(0.0, -1.0, 0.0),
        }
    }
    pub fn get_world_size(&self) -> f32 {
        self.world_size.x
    }

    pub fn set_cohesion(&mut self, c: f32) {
        self.cohesion = c;
    }

    pub fn set_matching(&mut self, m: f32) {
        self.matching = m;
    }

    pub fn set_avoid(&mut self, a: f32) {
        self.avoid = a;
    }
    pub fn clone_boids(&self) -> Boids {
        self.boids.clone()
    }
    pub fn set_next_boids(&mut self, next: Boids) {
        self.boids = next;
    }

    pub fn interpolate(&mut self, previous_state: Boids, alpha: f32) -> Boids {
        for (i, b) in self.boids.iter_mut().enumerate() {
            b.vel = (previous_state[i].vel).lerp(&b.vel, alpha);
            b.pos = (previous_state[i].pos).lerp(&b.pos, alpha);
        }
        self.boids.clone()
    }

    pub fn update_boids(&mut self, delta: f32) {
        let mut next = self.boids.clone();
        for (i, b) in next.iter_mut().enumerate() {
            let mut avoid_v = Vector3::zeros();
            let mut avg_v = Vector3::zeros();
            let mut avg_p = Vector3::zeros();
            let mut count = 0;
            for boid in self.boids.iter() {
                if boid.id == b.id {
                    continue;
                }

                let dist = b.pos.metric_distance(&boid.pos);
                if dist < self.protected_range {
                    avoid_v += b.pos - boid.pos;
                }
                if dist < self.visual_range {
                    avg_v += boid.vel;
                    avg_p += boid.pos;
                    count += 1;
                }
            }

            avg_v = avg_v.scale(1.0 / (count as f32));
            avg_p = avg_p.scale(1.0 / (count as f32));
            b.vel += avoid_v.scale(self.avoid)
                + (avg_v - b.vel).scale(self.matching)
                + (avg_p - b.pos).scale(self.cohesion);
            if b.pos.x < -self.world_size.x / 2.0 {
                b.vel.x += self.turn_factor;
            }
            if b.pos.y < -self.world_size.y / 2.0 {
                b.vel.y += self.turn_factor;
            }

            if b.pos.z < -self.world_size.z / 2.0 {
                b.vel.z += self.turn_factor;
            }
            if b.pos.x > self.world_size.x / 2.0 {
                b.vel.x -= self.turn_factor;
            }
            if b.pos.y > self.world_size.y / 2.0 {
                b.vel.y -= self.turn_factor;
            }

            if b.pos.z > self.world_size.z / 2.0 {
                b.vel.z -= self.turn_factor;
            }
            if i % 2 == 0 {
                b.vel.x = b.vel.x * (1.0 - self.bias) + (1.0 * self.bias);
            } else {
                b.vel.x = b.vel.x * (1.0 - self.bias) + (-1.0 * self.bias);
            }

            if b.vel.magnitude() < self.min_speed {
                b.vel.set_magnitude(self.min_speed);
            } else {
                b.vel = b.vel.cap_magnitude(self.max_speed);
            }
            b.pos += b.vel.scale(delta);
        }
        self.boids = next;
    }
}
