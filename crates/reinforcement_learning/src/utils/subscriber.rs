pub trait Subscriber<T> {
    fn consume(&mut self, data: T);
}
