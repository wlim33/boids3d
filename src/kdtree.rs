use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Clone)]
pub struct KDTreeNode {
    x: u32,
    y: u32,
    left: Option<Rc<RefCell<KDTreeNode>>>,
    right: Option<Rc<RefCell<KDTreeNode>>>,
}
impl KDTreeNode {
    pub fn new(x: u32, y: u32) -> Self {
        KDTreeNode {
            x,
            y,
            left: Option::None,
            right: Option::None,
        }
    }
}
