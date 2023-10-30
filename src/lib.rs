//! # Human Friendly Shell Args
//! [`lex`] and [`parse`] a [`str`] into shell-like [`args`] with simpler semantics.
//!
//! # Examples
//! hfsa will separate arguments similar to how a shell does and respect quotes, and escape
//! sequences.
//!
//! ```
//! let args = hfsa::args("git commit -m 'my commit message'")?;
//! assert_eq!(args, ["git", "commit", "-m", "my commit message"]);
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Semantics
//! - whitespace followed by a quote will open the quote, the same unescaped quote followed by
//!   whitespace closes it again
//!   - opening and closing quotes are discarded
//!   - tokens inside quotes are collected into a single arg
//! - preceding a token with `\` intorduces an escape sequence
//!   - regular args fail on escapes
//!   - single quotes ignore escapes
//!   - double quotes interpret escapes
//! - outside of quotes whitespace is discarded, inside it is kept
//!
//! ### Escape sequences
//! - `\n`, `\r` and `\t` for newline, carriage return and tab respectively
//! - `\"` for literal opening quotes
//! - `\\` for the escape character itself
//!
//! ## Goals
//! - no dependencies
//! - minimal clones
//! - lazy lexing and parsing
//! - efficient single pass parsing with minimal lookahead
//! - sensible human friendly parsing behavaior
//!   - `"don't do this"` is parsed into `["don't", "do", "this"]` instead of complaining about an
//!     unclosed quote
//! - span information for error reporting
//!
//! ## Non-goals
//! - variable interpolation
//! - evaluation of code

// TODO: glue contigous tokens back together in tmp to avoid clones where they could be borrowed

use std::borrow::Cow;
use std::error::Error;
use std::fmt::Display;
use std::iter::FusedIterator;
use std::num::NonZeroUsize;
use std::ops::Range;

/// A token representing non-empty part of the input and its kind.
#[derive(Debug, Clone, Copy)]
pub struct Token<'s> {
    lexeme: &'s str,
    start: usize,
    kind: TokenKind,
}

/// The semantic kind of a [`Token`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    /// A single quote `'`.
    Single,

    /// A double quote `"`.
    Double,

    /// A backslash for escaping `\`.
    Escape,

    /// Text.
    Text,

    /// Whitespace, one or more ASCII whitespace characters (` `, `\r`, `\n`, `\t`).
    Space,
}

impl<'s> Token<'s> {
    /// Create a new token from the given lexeme, start index and token kind, returns [`None`] if
    /// the lexeme is not a single kind.
    ///
    /// # Examples
    /// ```
    /// # fn _test() -> Option<()> {
    /// use hfsa::{Token, TokenKind};
    /// let token = Token::new("token", 0)?;
    /// assert_eq!(token.kind(), TokenKind::Text);
    /// # Some(()) } _test().unwrap();
    /// ```
    pub fn new(lexeme: &'s str, start: usize) -> Option<Self> {
        let (mut token, rest) = Self::lex(lexeme)?;
        token.start = start;
        if !rest.is_empty() {
            None
        } else {
            Some(token)
        }
    }

    /// Creates a new token wihtout validating it's invariants.
    ///
    /// # Safety
    /// The user must ensure that the lexeme is not empty and matches the kind. Constructing an
    /// invalid token causes no undefined behavior, but may result in panics when using it.
    ///
    /// # Examples
    /// ```
    /// use hfsa::{Token, TokenKind};
    /// let _ = unsafe { Token::new_unchecked("token", 0, TokenKind::Text) };
    /// ```
    #[inline]
    pub unsafe fn new_unchecked(lexeme: &'s str, start: usize, kind: TokenKind) -> Self {
        Self {
            lexeme,
            start,
            kind,
        }
    }

    /// The lexeme of this token, i.e. the textual representation. A lexeme is never empty.
    #[inline]
    pub fn lexeme(&self) -> &'s str {
        self.lexeme
    }

    /// The kind of this token.
    #[inline]
    pub fn kind(&self) -> TokenKind {
        self.kind
    }

    /// The start of this token's span, see [`Token::span`].
    #[inline]
    pub fn start(&self) -> usize {
        self.start
    }

    /// The length of this token's lexeme in bytes.
    #[inline]
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.lexeme.len()).expect("no empty token exists")
    }

    /// The span of this token, i.e. it's position relative to the input it was lexed from.
    ///
    /// # Examples
    /// ```
    /// # fn _test() -> Option<()> {
    /// use hfsa::Token;
    /// let input = "token";
    /// let (token, rest) = Token::lex(input)?;
    /// assert_eq!(&input[token.span()], token.lexeme());
    /// # Some(()) } _test().unwrap();
    /// ```
    #[inline]
    pub fn span(&self) -> Range<usize> {
        self.start..self.start + self.lexeme.len()
    }

    /// Split this token into two, with the second starting at the given index, returns [`None`] if
    /// one of the resulting tokens would be empty.
    ///
    /// # Panics
    /// Panics if the index is not on a UTF-8 code point boundary.
    ///
    /// # Examples
    /// ```
    /// # fn _test() -> Option<()> {
    /// use hfsa::Token;
    /// let input = "token";
    /// let (token, _) = Token::lex(input)?;
    /// let (first, second) = token.split(1)?;
    /// assert_eq!(first.lexeme(), "t");
    /// assert_eq!(second.lexeme(), "oken");
    /// # Some(()) } _test().unwrap();
    /// ```
    pub fn split(self, index: usize) -> Option<(Self, Self)> {
        if index == 0 || index >= self.lexeme.len() {
            return None;
        }

        let (first, second) = self.lexeme.split_at(index);
        Some((
            Self {
                lexeme: first,
                start: self.start,
                kind: self.kind,
            },
            Self {
                lexeme: second,
                start: self.start + index,
                kind: self.kind,
            },
        ))
    }

    fn break_esc_seq(self) -> (Self, Option<Self>) {
        let idx = self
            .lexeme
            .chars()
            .next()
            .expect("token cannot be empty")
            .len_utf8();

        match self.split(idx) {
            Some((f, s)) => (f, Some(s)),
            None => (self, None),
        }
    }
}

impl TokenKind {
    /// The single quote token literal.
    pub const SINGLE: &str = "'";

    /// The double quote token literal.
    pub const DOUBLE: &str = "\"";

    /// The escape token literal.
    pub const ESCAPE: &str = "\\";

    /// Whether this is [`TokenKind::Single`].
    #[inline]
    pub fn is_singe(self) -> bool {
        matches!(self, TokenKind::Single)
    }

    /// Whether this is [`TokenKind::Double`].
    #[inline]
    pub fn is_double(self) -> bool {
        matches!(self, TokenKind::Double)
    }

    /// Whether this is [`TokenKind::Escape`].
    #[inline]
    pub fn is_escape(self) -> bool {
        matches!(self, TokenKind::Escape)
    }

    /// Whether this is [`TokenKind::Text`].
    #[inline]
    pub fn is_text(self) -> bool {
        matches!(self, TokenKind::Text)
    }

    /// Whether this is [`TokenKind::Space`].
    #[inline]
    pub fn is_space(self) -> bool {
        matches!(self, TokenKind::Space)
    }
}

impl<'s> Token<'s> {
    /// Attempts to lex a single [`Token`] from the start of the given input, returning [`None`] if
    /// it is empty or the token and the remaining input.
    ///
    /// # Examples
    /// ```
    /// # fn _test() -> Option<()> {
    /// use hfsa::Token;
    /// let input = "git commit -m 'my commit message'";
    /// let (first, rest) = Token::lex(input)?;
    /// let (second, rest) = Token::lex(rest)?;
    /// let (third, rest) = Token::lex(rest)?;
    /// assert_eq!(first.lexeme(), "git");
    /// assert_eq!(second.lexeme(), " ");
    /// assert_eq!(third.lexeme(), "commit");
    /// # Some(()) } _test().unwrap();
    /// ```
    pub fn lex(input: &'s str) -> Option<(Token<'s>, &'s str)> {
        let ctor = |input: &'s str, len, kind| {
            let (lexeme, rest) = input.split_at(len);
            (
                Token {
                    lexeme,
                    start: 0,
                    kind,
                },
                rest,
            )
        };

        let bytes = input.as_bytes();
        Some(match *bytes.first()? {
            b'\'' => ctor(input, 1, TokenKind::Single),
            b'"' => ctor(input, 1, TokenKind::Double),
            b'\\' => ctor(input, 1, TokenKind::Escape),
            x => {
                type Ctor<'s> = fn(&'s str) -> Token<'s>;
                type Check = fn(u8) -> bool;

                let (ctor, check): (Ctor, Check) = if matches!(x, b' ' | b'\r' | b'\n' | b'\t') {
                    (
                        |token| Token {
                            lexeme: token,
                            kind: TokenKind::Space,
                            start: 0,
                        },
                        |b| !matches!(b, b' ' | b'\r' | b'\n' | b'\t'),
                    )
                } else {
                    (
                        |token| Token {
                            lexeme: token,
                            kind: TokenKind::Text,
                            start: 0,
                        },
                        |b| matches!(b, b'\'' | b'"' | b'\\' | b' ' | b'\r' | b'\n' | b'\t'),
                    )
                };

                if let Some(next) = bytes[1..].iter().position(|&b| check(b)) {
                    let (token, rest) = input.split_at(next + 1);
                    (ctor(token), rest)
                } else {
                    (ctor(input), "")
                }
            }
        })
    }
}

/// An [`Iterator`] over the [`Token`]s of a given input.
///
/// This struct is returned by the [`lex`] function, see its documentaion for more info.
#[derive(Debug, Clone)]
pub struct TokenIter<'s> {
    rest: &'s str,
    index: usize,
}

impl<'s> TokenIter<'s> {
    /// The rest of the input which was not yet tokenized.
    pub fn rest(&self) -> &'s str {
        self.rest
    }

    /// Returns the current index, that is the starting index of [`TokenIter::rest`] in the original
    /// input.
    pub fn index(&self) -> usize {
        self.index
    }
}

impl<'s> Iterator for TokenIter<'s> {
    type Item = Token<'s>;

    fn next(&mut self) -> Option<Self::Item> {
        let (mut token, rest) = Token::lex(self.rest)?;
        token.start += self.index;
        self.rest = rest;
        self.index += token.len().get();

        Some(token)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            if self.rest.is_empty() { 0 } else { 1 },
            Some(self.rest.len()),
        )
    }
}

impl FusedIterator for TokenIter<'_> {}

/// Yield the [`Token`]s in the given input.
///
/// # Example
/// ```
/// use hfsa::TokenKind::*;
/// let input = "git commit -m 'my commit message'";
/// let tokens: Vec<_> = hfsa::lex(input).map(|t | (t.lexeme(), t.kind())).collect();
/// assert_eq!(tokens, [
///     ("git",     Text),
///     (" ",       Space),
///     ("commit",  Text),
///     (" ",       Space),
///     ("-m",      Text),
///     (" ",       Space),
///     ("'",       Single),
///     ("my",      Text),
///     (" ",       Space),
///     ("commit",  Text),
///     (" ",       Space),
///     ("message", Text),
///     ("'",       Single),
/// ]);
/// ```
pub fn lex(input: &str) -> TokenIter<'_> {
    TokenIter {
        rest: input,
        index: 0,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Open(bool, usize),
    Normal,
}

impl State {
    fn fail_if_open(self) -> Result<(), ParseArgsError> {
        match self {
            State::Open(false, idx) => Err(ParseArgsError::UnclosedSingle(idx)),
            State::Open(true, idx) => Err(ParseArgsError::UnclosedDouble(idx)),
            _ => Ok(()),
        }
    }
}

/// An error returned by [`parse`] and [`args`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseArgsError {
    /// The input ended before a single quote was closed, contains the index of the opened single
    /// quote.
    UnclosedSingle(usize),

    /// The input ended before a double quote was closed, contains the index of the opened double
    /// quote.
    UnclosedDouble(usize),

    /// The input contained an unknown or invalid escape sequence, contains the index at which the
    /// escape sequence was found.
    InvalidEscape(usize),
}

impl Display for ParseArgsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnclosedSingle(idx) => write!(f, "unclosed single quote (opened at {idx}))"),
            Self::UnclosedDouble(idx) => write!(f, "unclosed double quote (opened at {idx})"),
            Self::InvalidEscape(idx) => write!(f, "invalid escape (at {idx})"),
        }
    }
}

impl Error for ParseArgsError {}

/// Parses [`Token`]s into arguments, automatically discarding quotes and interpreting escape
/// sequences. See the [module level documentation][self] for more info.
///
/// # Errors
/// Returns a [`ParseArgsError`] if the input was malformed:
/// - unclosed quotes
/// - trailing or invalid escape sequences
///
/// # Examples
/// ```
/// let input = "git commit -m 'my commit message'";
/// let args = hfsa::parse(hfsa::lex(input))?;
/// assert_eq!(args, vec!["git", "commit", "-m", "my commit message"]);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub fn parse<'s, I>(tokens: I) -> Result<Vec<Cow<'s, str>>, ParseArgsError>
where
    I: Iterator<Item = Token<'s>>,
{
    let mut tokens = tokens.peekable();
    let mut args: Vec<Cow<str>> = vec![];

    let mut tmp: Vec<Cow<str>> = vec![];
    let mut last = TokenKind::Space;
    let mut state = State::Normal;

    fn escape_next<'s>(next: Token<'s>, tmp: &mut Vec<Cow<'s, str>>) -> Option<()> {
        match next.kind {
            TokenKind::Double | TokenKind::Escape => {
                tmp.push(next.lexeme.into());
            }
            TokenKind::Text => {
                if !next.lexeme.starts_with(['n', 'r', 't']) {
                    return None;
                }

                let esc = match &next.lexeme[..1] {
                    "n" => "\n",
                    "r" => "\r",
                    "t" => "\t",
                    _ => unreachable!(),
                };

                tmp.push(esc.into());

                if let (_, Some(rest)) = next.break_esc_seq() {
                    tmp.push(rest.lexeme.into());
                }
            }
            TokenKind::Single | TokenKind::Space => return None,
        };

        Some(())
    }

    loop {
        let Some(token) = tokens.next() else {
            state.fail_if_open()?;
            break;
        };

        match state {
            State::Open(is_double, _) => {
                let mut f = |tokens: &mut std::iter::Peekable<I>| match tokens.next() {
                    Some(Token {
                        kind: TokenKind::Space,
                        ..
                    })
                    | None => {
                        args.push(String::from_iter(std::mem::take(&mut tmp)).into());
                        state = State::Normal;
                    }
                    Some(next) => {
                        tmp.push(token.lexeme.into());
                        tmp.push(next.lexeme.into());
                    }
                };

                match token.kind() {
                    TokenKind::Single if !is_double => f(&mut tokens),
                    TokenKind::Double if is_double => f(&mut tokens),
                    TokenKind::Escape if is_double => tokens
                        .next()
                        .and_then(|next| escape_next(next, &mut tmp))
                        .ok_or(ParseArgsError::InvalidEscape(token.start))?,
                    _ => {
                        tmp.push(token.lexeme.into());
                    }
                }
            }
            State::Normal => match token.kind() {
                TokenKind::Single | TokenKind::Double => {
                    if let Some(next) = tokens.peek().copied() {
                        if next.kind == token.kind && last == TokenKind::Space {
                            args.push(String::new().into());
                            let _ = tokens.next();
                            continue;
                        }
                    };

                    let new = if token.kind() == TokenKind::Single {
                        State::Open(false, token.start)
                    } else {
                        State::Open(true, token.start)
                    };

                    if last == TokenKind::Space {
                        state = new;
                    } else {
                        tmp.push(token.lexeme.into());
                    }
                }
                TokenKind::Escape => return Err(ParseArgsError::InvalidEscape(token.start)),
                TokenKind::Text => tmp.push(token.lexeme.into()),
                TokenKind::Space if last == TokenKind::Space => {
                    // NOTE: this happens only if there is leading space, so we don't push it
                    continue;
                }
                TokenKind::Space => {
                    args.push(String::from_iter(std::mem::take(&mut tmp)).into());
                }
            },
        }

        last = token.kind();
    }

    if !tmp.is_empty() {
        args.push(String::from_iter(tmp).into());
    }

    Ok(args)
}

/// Lex and parse the given input into its arguments. This is a shorthand for `parse(lex(input))`.
///
/// # Examples
/// ```
/// let args = hfsa::args("git commit -m 'my commit message'")?;
/// assert_eq!(args, ["git", "commit", "-m", "my commit message"]);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub fn args(input: &str) -> Result<Vec<Cow<str>>, ParseArgsError> {
    parse(lex(input))
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! tuple {
        ($kind:ident) => {
            (
                TokenKind::$kind,
                match TokenKind::$kind {
                    TokenKind::Single => "'",
                    TokenKind::Double => "\"",
                    TokenKind::Escape => "\\",
                    _ => unreachable!(),
                },
            )
        };
        ($kind:ident = $output:expr) => {
            (TokenKind::$kind, $output)
        };
    }

    macro_rules! test_lex {
        (
            $input:literal => [$($kind:ident $(= $output:expr)?),+ $(,)?]
        ) => {{
            let expected = vec![$(tuple!($kind $(= $output)?)),+];

            assert_eq!(
                lex($input).into_iter().map(|t| (t.kind, t.lexeme)).collect::<Vec<_>>(),
                expected,
            )
        }};
    }

    macro_rules! test_parse {
        ($input:literal => [$($output:expr),+ $(,)?]) => {
            assert_eq!(parse(lex($input)).unwrap(), vec![$($output),+])
        };
        ($input:literal => err $variant:ident($($fields:expr),+)) => {
            assert_eq!(parse(lex($input)), Err(ParseArgsError::$variant($($fields),+)))
        };
    }

    mod token {
        use super::*;

        #[test]
        fn split_first_empty() {
            let (token, _) = Token::lex("token").unwrap();
            assert!(token.split(0).is_none());
        }

        #[test]
        fn split_second_empty() {
            let (token, _) = Token::lex("token").unwrap();
            assert!(token.split(5).is_none());
        }
    }

    mod lex {
        use super::*;

        #[test]
        fn simple() {
            test_lex!(r#"  a b cd   "# => [
                Space = "  ",
                Text  = "a",
                Space = " ",
                Text  = "b",
                Space = " ",
                Text  = "cd",
                Space = "   ",
            ]);
            test_lex!(r#"abcd  efgh    ijk"# => [
                Text  = "abcd",
                Space = "  ",
                Text  = "efgh",
                Space = "    ",
                Text  = "ijk",
            ])
        }

        #[test]
        fn quotes() {
            test_lex!(r#"abc "def' hij"# => [
                Text  = "abc",
                Space = " ",
                Double,
                Text  = "def",
                Single,
                Space = " ",
                Text  = "hij",
            ]);
            test_lex!(r#""abc ""d'ef' hij'"# => [
                Double,
                Text  = "abc",
                Space = " ",
                Double,
                Double,
                Text  = "d",
                Single,
                Text  = "ef",
                Single,
                Space = " ",
                Text  = "hij",
                Single,
            ]);
        }

        #[test]
        fn escape() {
            test_lex!(r#"\"# => [
                Escape,
            ]);
            test_lex!(r#"\\ Wor\d"# => [
                Escape,
                Escape,
                Space = " ",
                Text  = "Wor",
                Escape,
                Text  = "d"
            ]);
        }

        #[test]
        fn other_whitespace() {
            test_lex!("a\nb \t " => [
                Text  = "a",
                Space = "\n",
                Text  = "b",
                Space = " \t "
            ]);
        }
    }

    mod parse {
        use super::*;

        #[test]
        fn simple() {
            test_parse!(r#"a b"# => ["a",  "b"]);
            test_parse!(r#"a \b"# => err InvalidEscape(2));
            test_parse!(r#"a \ b"# => err InvalidEscape(2));
        }

        #[test]
        fn single() {
            test_parse!(r#"a '' b"# => ["a", "", "b"]);
            test_parse!(r#"a b ''"# => ["a", "b", ""]);
            test_parse!(r#"'' a b"# => ["", "a", "b"]);
            test_parse!(r#"a 'b"# => err UnclosedSingle(2));
        }

        #[test]
        fn single_nested() {
            test_parse!(r#"'a '' b'"# => ["a '' b"]);
            test_parse!(r#"'a "" b'"# => ["a \"\" b"]);
        }

        #[test]
        fn double_simple() {
            test_parse!(r#"a "" b"# => ["a", "", "b"]);
            test_parse!(r#"a b """# => ["a", "b", ""]);
            test_parse!(r#""" a b"# => ["", "a", "b"]);
            test_parse!(r#"a "b"# => err UnclosedDouble(2));
        }

        #[test]
        fn double_nested() {
            test_parse!(r#""a '' b""# => ["a '' b"]);
            test_parse!(r#""a "" b""# => ["a \"\" b"]);
        }

        #[test]
        fn double_escape() {
            test_parse!(r#""a \"\" b""# => ["a \"\" b"]);
            test_parse!(r#""a \\ b""# => ["a \\ b"]);
            test_parse!(r#""a \ b""# => err InvalidEscape(3));
            test_parse!(r#"a "\""# => err UnclosedDouble(2));
            test_parse!(r#""a \"# => err InvalidEscape(3));
        }

        #[test]
        fn quotes_as_text() {
            test_parse!(r#"don't do this"# => ["don't", "do", "this"]);
            test_parse!(r#"a b'"# => ["a", "b'"]);
            test_parse!(r#"a b""# => ["a", "b\""]);
        }

        #[test]
        fn escape() {
            test_parse!(r#""\n""# => ["\n"]);
            test_parse!(r#""\na""# => ["\na"]);
            test_parse!(r#""\n a""# => ["\n a"]);
            test_parse!(r#""\"\n""# => ["\"\n"]);
            test_parse!(r#""\'\n""# => err InvalidEscape(1));
            test_parse!(r#"\\"# => err InvalidEscape(0));
            test_parse!(r#"a\\b"# => err InvalidEscape(1));
        }
    }
}
