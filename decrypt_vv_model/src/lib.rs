use std::fmt::{Debug, Display};

use cxx::CxxString;
use derive_more::Display;

#[cxx::bridge]
mod ffi {
    #[namespace = "decrypt_vv_model"]
    extern "Rust" {
        fn decrypt(content: &[u8], opts: &CxxString) -> Result<Vec<u8>>;
    }
}

fn decrypt(content: &[u8], opts: &CxxString) -> Result<Vec<u8>, ErrorMessage> {
    let opts = opts.to_str()?;
    if opts.is_empty() {
        Ok(content.to_owned())
    } else {
        let n = opts.parse::<u8>()?;
        Ok(nrot::rot(nrot::Mode::Decrypt, content, n))
    }
}

#[derive(Display)]
struct ErrorMessage(String);

impl<T: Display + Debug /* for the orphan rule */> From<T> for ErrorMessage {
    fn from(err: T) -> Self {
        Self(err.to_string())
    }
}
