//! # Endofunctor composition via the Co-Yoneda Lemma
//!
//! ## Functors in Rust
//!
//! Let's implement functors in Rust!
//!
//! Working around the lack of higher-kinded types, our trait for a functor
//! will look something like this:
//!
//! ```
//! pub trait Functor<A, B> {
//!    type Output;
//!    fn fmap<F: Fn(A) -> B>(self, F) -> Self::Output;
//! }
//! ```
//!
//! This works great as long as we write functions that take specific
//! types which are functors, but it is not possible to write a function
//! operating on a generic functor type and using `fmap` more than once.
//! For example, the following will not compile:
//!
//! ```
//! fn add_and_to_string<F: Functor<i32, String>>(x: F) -> F::Output {
//!    x.fmap(|n: i32| {
//!        n + 1
//!    }).fmap(|n: i32| {
//!        n.to_string()
//!    })
//! }!
//! ```
//!
//! While functors in general can be encoded to some extend
//! in Rust's trait system, what we usually mean when we
//! say "Functor" is a covariant endofunctor.
//! An endofunctor is a functor that maps back to the same category,
//! e.g. it maps a function between `A` and `B` to a function between
//! `Box<A>` and `Box<B>`, not between `Box<A>` and `Option<B>`.
//!
//! Especially when looking at functor composition, it is useful to
//! be able to encode endofunctors, because it allows us to chain
//! multiple calls to `fmap`, knowing that the result is also a functor,
//! and can be `fmap`'ed further. In Rust this is not possible,
//! because of a lack of higher-kinded types.
//!
//! ## The Co-Yoneda Lemma
//!
//! Let's define a data type called `Coyoneda`:
//!
//! ```
//! Coyoneda :: (b -> a) -> f b -> Coyoneda f a
//! ```
//!
//! This datatype is a functor, which uses function composition
//! to accumulate the mapping function, without changing the captured
//! `f b`. The implementation for `fmap` is trivial:
//!
//! ```
//! fmap f (Coyoneda g a) = Coyoneda (f . g) a
//! ```
//!
//! The co-yoneda lemma states that for a covariant functor `f`,
//! this `Coyoneda f` is naturally isomorphic to `f`.
//! Practically speaking, this means that we can lift any `f a` into a `Coyoneda f a`,
//! and given a function `(a -> b) -> f b`, we can retrieve back a `f a` from a `Coyoneda f a`.
//!
//! ## Composing Coyoneda
//!
//! Now here's the catch: Since we have a parameterized datatype that is isomorphic to any functor,
//! we can lift functors into Coyoneda to compose them safely within Rust's type system!
//!
//! For example, let's implement a function that is generic for any functor:
//!
//! ```
//! fn add_and_to_string<F: Functor<i32, String>>(x: F) -> F::Output {
//!    Coyoneda::from(x).map(|n: i32| {
//!        n + 1
//!    }).map(|n: i32| {
//!        n.to_string()
//!    }).unwrap()
//! }!
//! ```
//!
//! Given we implemented a functor for `Option`, we can now do the following:
//!
//! ```
//! let y = add_and_to_string(Some(42));
//! assert_eq!(y, Some("43".to_string()))
//! ```
//!
//! ... or for `Box`:
//!
//! ```
//! let y = add_and_to_string(Box::new(42));
//! assert_eq!(y, Box::new("43".to_string()))
//! ```
//!
//! ... and for every other functor as well. Yay!

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

    pub fn unwrap(self) -> <T as Functor<A, B>>::Output where T: Functor<A, B> {
        T::fmap(self.point, self.morph)
    }

}

impl<'a, T, A> From<T> for Coyoneda<'a, T, A, A> {
    fn from(x: T) -> Coyoneda<'a, T, A, A> {
        Coyoneda{point: x, morph: Morphism::new()}
    }
}

pub trait Functor<A, B> {
    type Output;
    fn fmap<F: Fn(A) -> B>(self, F) -> Self::Output;
}

impl<A, B> Functor<A, B> for Box<A> {
    type Output = Box<B>;
    fn fmap<F: Fn(A) -> B>(self, f: F) -> Self::Output {
        Box::new(f(*self))
    }
}

impl<A, B> Functor<A, B> for Option<A> {
    type Output = Option<B>;
    fn fmap<F: Fn(A) -> B>(self, f: F) -> Self::Output {
        Option::map(self, f)
    }
}

impl<A, B, E> Functor<A, B> for Result<A, E> {
    type Output = Result<B, E>;
    fn fmap<F: Fn(A) -> B>(self, f: F) -> Self::Output {
        Result::map(self, f)
    }
}

mod test {
#![cfg(test)]

    use super::*;

    fn add_and_to_string<F: Functor<i32, String>>(x: F) -> F::Output {
        Coyoneda::from(x).map(|n: i32| {
            n + 1
        }).map(|n: i32| {
            n.to_string()
        }).unwrap()
    }

    #[test]
    fn test_box() {
        let x = Box::new(42);
        let y = add_and_to_string(x);
        assert_eq!(y, Box::new("43".to_string()))
    }

    #[test]
    fn test_option() {
        let x = Some(42);
        let y = add_and_to_string(x);
        assert_eq!(y, Some("43".to_string()))
    }

    #[test]
    fn test_result() {
        let x: Result<i32, ()> = Ok(42);
        let y = add_and_to_string(x);
        assert_eq!(y, Ok("43".to_string()))
    }

}
