extern crate morphism;

use morphism::Morphism;

pub struct Coyoneda<'a, T, A, B> {
    point: T,
    morph: Morphism<'a, A, B>
}

impl<'a, T, A, B> Coyoneda<'a, T, A, B> {

    pub fn map<C, F: Fn(B) -> C + 'a>(self, f: F) -> Coyoneda<'a, T, A, C> {
        Coyoneda{point: self.point, morph: self.morph.tail(f)}
    }

    pub fn lower(self) -> <T as Map<A, B>>::Output where T: Map<A, B> {
        T::map(self.point, self.morph)
    }

}

impl<'a, T, A> From<T> for Coyoneda<'a, T, A, A> {
    fn from(x: T) -> Coyoneda<'a, T, A, A> {
        Coyoneda{point: x, morph: Morphism::new()}
    }
}

pub trait Map<A, B> {
    type Output;
    fn map<F: Fn(A) -> B>(self, F) -> Self::Output;
}

impl<A, B> Map<A, B> for Box<A> {
    type Output = Box<B>;
    fn map<F: Fn(A) -> B>(self, f: F) -> Self::Output {
        Box::new(f(*self))
    }
}

impl<A, B> Map<A, B> for Option<A> {
    type Output = Option<B>;
    fn map<F: Fn(A) -> B>(self, f: F) -> Self::Output {
        Option::map(self, f)
    }
}

impl<A, B, E> Map<A, B> for Result<A, E> {
    type Output = Result<B, E>;
    fn map<F: Fn(A) -> B>(self, f: F) -> Self::Output {
        Result::map(self, f)
    }
}

#[test]
fn test_box() {
    let x = Box::new(42);
    let y = Coyoneda::from(x);
    let z = y.map(|n: i32| {
        n.to_string()
    });
    assert_eq!(z.lower(), Box::new("42".to_string()))
}

#[test]
fn test_option() {
    let x = Some(42);
    let y = Coyoneda::from(x);
    let z = y.map(|n: i32| {
        n.to_string()
    });
    assert_eq!(z.lower(), Some("42".to_string()))
}

#[test]
fn test_result() {
    let x: Result<i32, ()> = Ok(42);
    let y = Coyoneda::from(x);
    let z = y.map(|n: i32| {
        n.to_string()
    });
    assert_eq!(z.lower(), Ok("42".to_string()))
}
