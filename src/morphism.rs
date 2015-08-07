//! This crate provides a structure for suspended closure composition.
//! Composition is delayed and executed in a loop when a `Morphism` is
//! applied to an argument.
//!
//! The motivation for `Morphism` is to provide a means of composing
//! and evaluating an unbounded (within heap constraints) number of
//! closures without blowing the stack. In other words, `Morphism` is
//! one way to work around the lack of tail-call optimization in Rust.

use std::collections::{
    LinkedList,
    VecDeque,
};
use std::marker::{
    PhantomData,
};
use std::mem::{
    transmute,
};

/// A suspended chain of closures that behave as a function from type
/// `A` to type `B`.
///
/// When `B = A` the parameter `B` can be omitted: `Morphism<'a, A>`
/// is equivalent to `Morphism<'a, A, A>`.  This is convenient for
/// providing annotations with `Morphism::new()`.
pub struct Morphism<'a, A, B = A> {
    mfns: LinkedList<VecDeque<Box<Fn(*const ()) -> *const () + 'a>>>,
    phan: PhantomData<(A, B)>,
}

#[allow(dead_code)]
enum Void {}
impl Morphism<'static, Void> {
    /// Create the identity chain.
    ///
    /// # Example
    ///
    /// ```rust
    /// use morphism::Morphism;
    ///
    /// assert_eq!(Morphism::new::<u64>()(42u64), 42u64);
    /// ```
    #[inline]
    pub fn new<'a, A>() -> Morphism<'a, A> {
        Morphism {
            mfns: {
                let mut mfns = LinkedList::new();
                mfns.push_back(VecDeque::new());
                mfns
            },
            phan: PhantomData,
        }
    }
}

#[allow(dead_code)]
impl<'a, B, C> Morphism<'a, B, C> {
    #[inline(always)]
    pub unsafe fn unsafe_push_front<A, F>(&mut self, f: F) -> ()
        where F: Fn(A) -> B + 'a,
    {
        match self {
            &mut Morphism {
                ref mut mfns,
                ..
            }
            => {
                // assert!(!mfns.is_empty())
                let head = mfns.front_mut().unwrap();
                let g = box move |ptr| {
                    transmute::<Box<B>, *const ()>(
                        box f.call((
                            *transmute::<*const (), Box<A>>(ptr)
                                ,))
                            )
                };
                head.push_front(g);
            },
        }
    }

    /// Attach a closure to the front of the closure chain. This corresponds to
    /// closure composition at the domain (pre-composition).
    ///
    /// # Example
    ///
    /// ```rust
    /// use morphism::Morphism;
    ///
    /// let f = Morphism::new::<Option<String>>()
    ///     .head(|x: Option<u64>| x.map(|y| y.to_string()))
    ///     .head(|x: Option<u64>| x.map(|y| y - 42u64))
    ///     .head(|x: u64| Some(x + 42u64 + 42u64));
    /// assert_eq!(f(0u64), Some("42".to_string()));
    /// ```
    #[inline]
    pub fn head<A, F>(self, f: F) -> Morphism<'a, A, C>
        where F: Fn(A) -> B + 'a,
    {
        let mut self0 = self;
        unsafe {
            (&mut self0).unsafe_push_front(f);
            transmute(self0)
        }
    }

    /// Mutate a given `Morphism<B, C>` by pushing a closure of type
    /// `Fn(B) -> B` onto the front of the chain.
    ///
    /// # Example
    ///
    /// ```rust
    /// use morphism::Morphism;
    ///
    /// let mut f = Morphism::new::<u64>();
    /// for i in (0..10u64) {
    ///     (&mut f).push_front(move |x| x + i);
    /// }
    /// assert_eq!(f(0u64), 45u64);
    /// ```
    #[inline]
    pub fn push_front<F>(&mut self, f: F) -> ()
        where F: Fn(B) -> B + 'a,
    {
        unsafe {
            self.unsafe_push_front(f)
        }
    }
}

#[allow(dead_code)]
impl<'a, A, B> Morphism<'a, A, B> {
    #[inline(always)]
    pub unsafe fn unsafe_push_back<C, F>(&mut self, f: F) -> ()
        where F: Fn(B) -> C + 'a,
    {
        match self {
            &mut Morphism {
                ref mut mfns,
                ..
            }
            => {
                // assert!(!mfns.is_empty())
                let tail = mfns.back_mut().unwrap();
                let g = box move |ptr| {
                    transmute::<Box<C>, *const ()>(
                        box f.call((
                            *transmute::<*const (), Box<B>>(ptr)
                        ,))
                    )
                };
                tail.push_back(g);
            },
        }
    }

    /// Attach a closure to the back of the closure chain. This corresponds to
    /// closure composition at the codomain (post-composition).
    ///
    /// # Example
    ///
    /// ```rust
    /// use morphism::Morphism;
    ///
    /// let f = Morphism::new::<u64>()
    ///     .tail(|x| Some(x + 42u64 + 42u64))
    ///     .tail(|x| x.map(|y| y - 42u64))
    ///     .tail(|x| x.map(|y| y.to_string()));
    /// assert_eq!(f(0u64), Some("42".to_string()));
    /// ```
    #[inline]
    pub fn tail<C, F>(self, f: F) -> Morphism<'a, A, C>
        where F: Fn(B) -> C + 'a,
    {
        let mut self0 = self;
        unsafe {
            (&mut self0).unsafe_push_back(f);
            transmute(self0)
        }
    }

    /// Mutate a given `Morphism<A, B>` by pushing a closure of type
    /// `Fn(B) -> B` onto the back of the chain.
    ///
    /// # Example
    ///
    /// ```rust
    /// use morphism::Morphism;
    ///
    /// let mut f = Morphism::new::<u64>();
    /// for i in (0..10u64) {
    ///     (&mut f).push_back(move |x| x + i);
    /// }
    /// assert_eq!(f(0u64), 45u64);
    /// ```
    #[inline]
    pub fn push_back<F>(&mut self, f: F) -> ()
        where F: Fn(B) -> B + 'a,
    {
        unsafe {
            self.unsafe_push_back(f)
        }
    }

    /// Compose one `Morphism` with another.
    ///
    /// # Example
    ///
    /// ```rust
    /// use morphism::Morphism;
    ///
    /// let mut f = Morphism::new::<u64>();
    /// for _ in (0..100000u64) {
    ///     f = f.tail(|x| x + 42u64);
    /// }
    /// // the type changes to Morphism<u64, Option<u64>> so rebind f
    /// let f = f.tail(|x| Some(x));
    ///
    /// let mut g = Morphism::new::<Option<u64>>();
    /// for _ in (0..99999u64) {
    ///     g = g.tail(|x| x.map(|y| y - 42u64));
    /// }
    /// // the type changes to Morphism<Option<u64>, String> so rebind g
    /// let g = g.tail(|x| x.map(|y| y + 1000u64).unwrap().to_string());
    ///
    /// assert_eq!(f.then(g)(0u64), "1042".to_string());
    /// ```
    #[inline]
    pub fn then<C>(self, mut other: Morphism<'a, B, C>) -> Morphism<'a, A, C> {
        match self {
            Morphism {
                mfns: mut lhs,
                ..
            }
            => {
                match other {
                    Morphism {
                        mfns: ref mut rhs,
                        ..
                    }
                    => {
                        Morphism {
                            mfns: {
                                lhs.append(rhs);
                                lhs
                            },
                            phan: PhantomData,
                        }
                    },
                }
            },
        }
    }

    /// Given an argument, run the chain of closures in a loop and return the
    /// final result.
    #[inline]
    fn run(&self, x: A) -> B { unsafe {
        let mut res = transmute::<Box<A>, *const ()>(box x);
        for fns in self.mfns.iter() {
            for f in fns.iter() {
                res = f.call((res,));
            }
        }
        *transmute::<*const (), Box<B>>(res)
    }}
}

impl<'a, A, B> FnOnce<(A,)> for Morphism<'a, A, B> {
    type Output = B;
    extern "rust-call" fn call_once(self, (x,): (A,)) -> B {
        self.run(x)
    }
}

impl<'a, A, B> FnMut<(A,)> for Morphism<'a, A, B> {
    extern "rust-call" fn call_mut(&mut self, (x,): (A,)) -> B {
        self.run(x)
    }
}

impl<'a, A, B> Fn<(A,)> for Morphism<'a, A, B> {
    extern "rust-call" fn call(&self, (x,): (A,)) -> B {
        self.run(x)
    }
}

#[cfg(test)]
mod tests
{
    use super::Morphism;

    #[test]
    fn readme() {
        let mut f = Morphism::new::<u64>();
        for _ in (0..100000u64) {
            f = f.tail(|x| x + 42u64);
        }

        let mut g = Morphism::new::<Option<u64>>();
        for _ in (0..99999u64) {
            g = g.tail(|x| x.map(|y| y - 42u64));
        }

        // type becomes Morphism<u64, (Option<u64>, bool, String)> so rebind g
        let g = g
            .tail(|x| (x.map(|y| y + 1000u64), "welp".to_string()))
            .tail(|(l, r)| (l.map(|y| y + 42u64), r))
            .tail(|(l, r)| (l, l.is_some(), r))
            .head(|x| Some(x));

        let h = f.then(g);

        assert_eq!(h(0u64), (Some(1084), true, "welp".to_string()));
        assert_eq!(h(1000u64), (Some(2084), true, "welp".to_string()));
    }

    #[test]
    fn fn_like() {
        let mut f = Morphism::new::<u64>();
        for _ in (0..10000) {
            f = f.tail(|x| x + 42);
        }

        // ::map treats f like any other Fn
        let res: u64 = (0..100).map(f).sum();

        assert_eq!(res, 42004950);
    }
}
