# human friendly shell arguments
This is a small library for lexing and parsing strings into shell like arguments.

```rust
let args = hfsa::args("git commit -m 'my commit message'")?;
assert_eq!(args, ["git", "commit", "-m", "my commit message"]);
```

This is primarily used in a discord bot I wrote for personal use.
