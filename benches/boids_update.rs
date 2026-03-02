use boids_3d_rs::World;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn boids_update_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("boids_update");
    for &count in &[128_usize, 512, 2048] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let mut world = World::new();
            world.generate_boids(count, 0);
            b.iter(|| world.boids_update(0.016));
        });
    }
    group.finish();
}

fn world_update_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("world_update");
    for &count in &[128_usize, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let mut world = World::new();
            world.generate_boids(count, 0);
            b.iter(|| world.update(0.016));
        });
    }
    group.finish();
}

fn calculate_render_pipeline_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("calculate_final_render_objects");
    group.bench_function("render_objects/512", |b| {
        let mut world = World::new();
        world.generate_boids(512, 0);
        b.iter(|| world.calculate_final_render_objects());
    });
    group.finish();
}

fn generate_boids_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_boids");
    for &count in &[64_usize, 256, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter(|| {
                let mut world = World::new();
                world.generate_boids(count, 0);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    boids_update_benchmark,
    world_update_benchmark,
    calculate_render_pipeline_benchmark,
    generate_boids_benchmark
);
criterion_main!(benches);
