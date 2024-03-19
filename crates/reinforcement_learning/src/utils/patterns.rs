use std::rc::Rc;

pub trait Subscriber<T> {
    fn consume(&mut self, data: T);
}

pub trait Publisher<T> {
    fn publish(&mut self, data: T);
    fn subscribe(&mut self, subscriber: Rc<dyn Subscriber<T>>);
}
