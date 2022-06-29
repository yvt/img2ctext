mod image;
mod quant;
pub use self::{image::*, quant::*};

/// The number of channels in a color.
pub const NUM_CHANNELS: usize = 3;

/// The number of colors in each palette.
pub const PALETTE_LEN: usize = 2;
