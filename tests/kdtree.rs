use wasm_boids::kdtree::KDTreeNode;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_adds_two() {
        let result = 1 + 2;
        let node = KDTreeNode::new(0, 0);

        assert_eq!(result, 3);
    }
}
