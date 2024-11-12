pub struct BoidsParameters {
    pub matching_factor: f32,
    pub avoid_factor: f32,
    pub cohesion_factor: f32,
    pub turn_factor: f32,
    pub visual_range: f32,
    pub protected_range: f32,
    pub max_speed: f32,
    pub min_speed: f32,
    //pub bias: f32,
    //pub max_bias: f32,
    //pub bias_increment: f32,
}
impl BoidsParameters {
    pub fn new() -> Self {
        BoidsParameters {
            matching_factor: 0.05,
            cohesion_factor: 0.0005,
            avoid_factor: 0.05,
            turn_factor: 0.2,
            protected_range: 8.0,
            visual_range: 40.0,
            //bias: 0.001,
            //bias_increment: 0.00004,
            //max_bias: 0.01,
            max_speed: 12.0,
            min_speed: 12.0,
        }
    }
}
