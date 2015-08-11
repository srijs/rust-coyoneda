//! # Functor composition via the Co-Yoneda Lemma
//!
//! ## Functors in Rust
//!
//! Let's implement functors in Rust!
//!
//! Working around the lack of higher-kinded types, our trait for a functor
//! will look something like this:
//!
//! ```
//! pub trait Functor<'a, A, B> {
//!    type Output;
//!    fn fmap<F: Fn(A) -> B + 'a>(self, F) -> Self::Output;
//! }
//! ```
//!
//! This works great as long as we write functions that take specific
//! types which are functors, but it is not possible to write a function
//! operating on a generic functor type and using `fmap` more than once.
//! For example, the following will not compile:
//!
//! ```
//! fn add_and_to_string<'a, F: Functor<'a, i32, String>>(x: F) -> F::Output {
//!    x.fmap(|n: i32| n + 1)
//!     .fmap(|n: i32| n.to_string())
//! }
//! ```
//!
//! While functors in general can be encoded to some extend
//! in Rust's trait system, what we can't encode for a lack of higher-kinded
//! types, is the fact that a functor `Box` maps a function between `A` and `B`
//! to a function between `Box<A>` and `Box<B>`, not between `Box<A>` and `Option<B>`.
//!
//! Especially when looking at functor composition, it is useful to
//! be able to encode this fact, because it allows us to chain
//! multiple calls to `fmap`, knowing that the result is also a functor,
//! and can be `fmap`'ed further.
//!
//! ## The Co-Yoneda Lemma
//!
//! Let's define a data type called `Coyoneda`:
//!
//! ```
//! pub struct Coyoneda<'a, T: Param, B> {
//!     point: T,
//!     morph: Fn(T::Param) -> B + 'a
//! }
//! ```
//!
//! This datatype is a functor, which uses function composition
//! to accumulate the mapping function, without changing the captured
//! `T`. The implementation for `Functor` is trivial:
//!
//! ```
//! impl<'a, T: Param, B, C> Functor<'a, B, C> for Coyoneda<'a, T, B> {
//!     type Output = Coyoneda<'a, T, C>;
//!     fn fmap<F: Fn(B) -> C + 'a>(self, f: F) -> Coyoneda<'a, T, C> {
//!         let g = self.morph;
//!         Coyoneda{point: self.point, morph: move |x| f(g(x))}
//!     }
//! }
//! ```
//!
//! The co-yoneda lemma states that for a covariant functor `f`,
//! this `Coyoneda f` is naturally isomorphic to `f`.
//! Practically speaking, this means that we can lift any `f a` into a `Coyoneda f a`,
//! and given a function `(a -> b) -> f b`, we can retrieve back a `f b` from a `Coyoneda f b`.
//!
//! ## Composing Coyoneda
//!
//! Now here's the catch: Since we have a parameterized datatype that is isomorphic to any functor,
//! we can lift functors into Coyoneda to compose them safely within Rust's type system!
//!
//! For example, let's implement a function that is generic for any functor,
//! by operating on our `Coyoneda` type:
//!
//! ```
//! fn add_and_to_string<T: Param>(y: Coyoneda<T, i32>) -> Coyoneda<T, String> {
//!    y.fmap(|n: i32| n + 1)
//!     .fmap(|n: i32| n.to_string())
//! }
//! ```
//!
//! Given we implemented a functor for `Option`, we can now do the following:
//!
//! ```
//! let y = add_and_to_string(From::from(Some(42)));
//! assert_eq!(y.unwrap(), Some("43".to_string()))
//! ```
//!
//! ... or for `Box`:
//!
//! ```
//! let y = add_and_to_string(From::from(Box::new(42)));
//! assert_eq!(y.unwrap(), Box::new("43".to_string()))
//! ```
//!
//! ... and for every other functor as well. Yay!

mod morphism;

use morphism::Morphism;

pub trait Param { type Param; }

pub trait Functor<'a, A, B>: Param {
    type Output;
    fn fmap<F: Fn(A) -> B + 'a>(self, F) -> Self::Output;
}

pub struct Coyoneda<'a, T: Param, B> {
    point: T,
    morph: Morphism<'a, T::Param, B>
}

impl<'a, T: 'a + Param, B: 'a> Coyoneda<'a, T, B> {

    pub fn transform<U: Param, F, G>(self, f: F, g: G) -> Coyoneda<'a, U, B>
        where F: Fn(T) -> U, G: 'a + Fn(U::Param) -> T::Param {
        let Coyoneda{point: t, morph: m} = self;
        Coyoneda{point: f(t), morph: m.head(g)}
    }

    pub fn unwrap(self) -> <T as Functor<'a, <T as Param>::Param, B>>::Output
        where T: Functor<'a, <T as Param>::Param, B>, <T as Param>::Param: 'a {
        let m = self.morph;
        T::fmap(self.point, move |a| { m.run(a) })
    }

}

impl<'a, T: Param, B> Param for Coyoneda<'a, T, B> {
    type Param = B;
}

impl<'a, T: Param, B, C> Functor<'a, B, C> for Coyoneda<'a, T, B> {
    type Output = Coyoneda<'a, T, C>;
    fn fmap<F: Fn(B) -> C + 'a>(self, f: F) -> Coyoneda<'a, T, C> {
        Coyoneda{point: self.point, morph: self.morph.tail(f)}
    }
}

impl<'a, T: Param> From<T> for Coyoneda<'a, T, <T as Param>::Param> {
    fn from(x: T) -> Coyoneda<'a, T, <T as Param>::Param> {
        Coyoneda{point: x, morph: Morphism::new()}
    }
}

impl<A> Param for Box<A> {
    type Param = A;
}

impl<'a, A, B> Functor<'a, A, B> for Box<A> {
    type Output = Box<B>;
    fn fmap<F: Fn(A) -> B>(self, f: F) -> Self::Output {
        Box::new(f(*self))
    }
}

impl<A> Param for Option<A> {
    type Param = A;
}

impl<'a, A, B> Functor<'a, A, B> for Option<A> {
    type Output = Option<B>;
    fn fmap<F: Fn(A) -> B>(self, f: F) -> Self::Output {
        Option::map(self, f)
    }
}

impl<A, E> Param for Result<A, E> {
    type Param = A;
}

impl<'a, A, B, E> Functor<'a, A, B> for Result<A, E> {
    type Output = Result<B, E>;
    fn fmap<F: Fn(A) -> B>(self, f: F) -> Self::Output {
        Result::map(self, f)
    }
}

mod test {
#![cfg(test)]

    use super::*;

    fn add_and_to_string<T: Param>(y: Coyoneda<T, i32>) -> Coyoneda<T, String> {
        y.fmap(|n: i32| n + 1)
         .fmap(|n: i32| n.to_string())
         .fmap(|s| s + "foo")
         .fmap(|s| s + "bar")
    }

    #[test]
    fn test_box() {
        let x = Box::new(42);
        let y = add_and_to_string(From::from(x)).unwrap();
        assert_eq!(y, Box::new("43foobar".to_string()))
    }

    #[test]
    fn test_option() {
        let x = Some(42);
        let y = add_and_to_string(From::from(x)).unwrap();
        assert_eq!(y, Some("43foobar".to_string()))
    }

    #[test]
    fn test_result() {
        let x: Result<i32, ()> = Ok(42);
        let y = add_and_to_string(From::from(x)).unwrap();
        assert_eq!(y, Ok("43foobar".to_string()))
    }

}
