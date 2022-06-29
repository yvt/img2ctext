use core::fmt;

use ndarray::{s, Array2, Array3, Array4, ArrayView2, Axis};

use crate::{NUM_CHANNELS, PALETTE_LEN};

/// A quantized image.
///
/// # Invariants
///
/// - All elements of `image_dim` and `cell_dim` must be positive values.
///   `[tag:qimage_non_empty]`
#[derive(Default, Debug, Clone)]
#[non_exhaustive]
pub struct QuantizedImage {
    /// `[image_dim[0] * cell_dim[0], image_dim[1] * cell_dim[1]]`
    pub indices: Array2<bool>,
    /// `[PALETTE_LEN, NUM_CHANNELS, image_dim[0], image_dim[1]]`
    pub palettes: Array4<u8>,
}

impl QuantizedImage {
    /// Get the number of cells in each dimension as `[height, width]`.
    #[inline]
    pub fn image_dim(&self) -> [usize; 2] {
        let (_, _, h, w) = self.palettes.dim();
        [h, w]
    }

    /// Get the number of pixels in each dimension of each cell as
    /// `[height, width]`.
    #[inline]
    pub fn cell_dim(&self) -> [usize; 2] {
        let (total_h, total_w) = self.indices.dim();
        let [image_h, image_w] = self.image_dim();
        debug_assert_eq!(total_w % image_w, 0);
        debug_assert_eq!(total_h % image_h, 0);
        [total_h / image_h, total_w / image_w]
    }

    /// Convert this image to a colored preformatted text using the glyph set
    /// [`img2text::GLYPH_SET_1X2`].
    ///
    /// # Panics
    ///
    /// This method will panic if `self.cell_dim() != [1, 2]`.
    pub fn write_1x2_to<W: WriteCell + ?Sized>(&self, out: &mut W) -> Result<(), W::Error> {
        let mut opts = img2text::Bmp2textOpts::default();
        opts.glyph_set = img2text::GLYPH_SET_1X2;
        self.write_to(&opts, [2, 1], out)
    }

    /// Convert this image to a colored preformatted text using the glyph set
    /// [`img2text::GLYPH_SET_2X2`].
    ///
    /// # Panics
    ///
    /// This method will panic if `self.cell_dim() != [2, 2]`.
    pub fn write_2x2_to<W: WriteCell + ?Sized>(&self, out: &mut W) -> Result<(), W::Error> {
        let mut opts = img2text::Bmp2textOpts::default();
        opts.glyph_set = img2text::GLYPH_SET_2X2;
        self.write_to(&opts, [2, 2], out)
    }

    /// Convert this image to a colored preformatted text using the glyph set
    /// [`img2text::GLYPH_SET_2X3`].
    ///
    /// # Panics
    ///
    /// This method will panic if `self.cell_dim() != [3, 2]`.
    pub fn write_2x3_to<W: WriteCell + ?Sized>(&self, out: &mut W) -> Result<(), W::Error> {
        let mut opts = img2text::Bmp2textOpts::default();
        opts.glyph_set = img2text::GLYPH_SET_2X3;
        self.write_to(&opts, [3, 2], out)
    }

    /// Convert this image to a colored preformatted text using the glyph set
    /// [`img2text::GLYPH_SET_SLC`].
    ///
    /// # Panics
    ///
    /// This method will panic if `self.cell_dim() != [3, 3]`.
    pub fn write_slc_to<W: WriteCell + ?Sized>(&self, out: &mut W) -> Result<(), W::Error> {
        let mut opts = img2text::Bmp2textOpts::default();
        opts.glyph_set = img2text::GLYPH_SET_SLC;
        self.write_to(&opts, [3, 3], out)
    }

    /// Convert this image to a colored preformatted text using an `img2text`
    /// glyph set.
    ///
    /// The glyph set must meet the following criteria:
    ///
    ///  - Its `mask_overlap` is `[0, 0]`.
    ///  - Its `mask_dims` is equal to `cell_dim` (note: the dimension order
    ///    is different between this crate and `img2text`).
    ///  - It produces one `char` per cell.
    ///
    fn write_to<W: WriteCell + ?Sized>(
        &self,
        opts: &img2text::Bmp2textOpts,
        cell_dim: [usize; 2],
        out: &mut W,
    ) -> Result<(), W::Error> {
        assert_eq!(self.cell_dim(), cell_dim);
        let image_dim = self.image_dim();

        // Uncolored line buffer
        let mut line = String::with_capacity(
            img2text::max_output_len_for_image_dims([self.image_dim()[1], cell_dim[0]], opts)
                .unwrap_or(0),
        );

        // Color buffer of size `[PALETTE_LEN, NUM_CHANNELS, image_dim[1]]` with
        // the layout identical to `[[RGB8; PALETTE_LEN]; image_dim[1]]`
        // (i.e., the standard layout of an array of size
        // `[image_dim[1], PALETTE_LEN, NUM_CHANNELS]`)
        let mut line_palettes =
            Array3::zeros((image_dim[1], PALETTE_LEN, NUM_CHANNELS)).permuted_axes((1, 2, 0));
        assert_eq!(
            line_palettes.dim(),
            (PALETTE_LEN, NUM_CHANNELS, image_dim[1])
        );

        // Working buffer for `img2text`
        let mut bmp2text = img2text::Bmp2text::new();

        let cell_rows_indices = self.indices.axis_chunks_iter(Axis(0), cell_dim[0]);
        let cell_rows_palettes = self
            .palettes
            .slice(s!(..PALETTE_LEN, ..NUM_CHANNELS, .., ..));
        let cell_rows_palettes = cell_rows_palettes.axis_iter(Axis(2));
        for (cell_row_indices, cell_row_palettes) in cell_rows_indices.zip(cell_rows_palettes) {
            // Output one uncolored line (with a line terminator) to `line`
            bmp2text
                .transform_and_write(
                    &ImageReadImpl {
                        indices: cell_row_indices,
                    },
                    opts,
                    &mut line,
                )
                .expect("`write_str` on `String` unexpectedly failed");

            // Copy this line's colors to `line_palettes`
            line_palettes.assign(&cell_row_palettes);
            let line_palettes = line_palettes
                .as_slice_memory_order()
                .unwrap()
                .chunks_exact(PALETTE_LEN * NUM_CHANNELS)
                .map(|channels| {
                    // FIXME: Use `array::from_fn` when it's stable
                    // FIXME: Use `array::array_chunks` when it's stable
                    let mut colors = channels.chunks_exact(NUM_CHANNELS);
                    [(); PALETTE_LEN].map(|()| {
                        rgb::RGB8::from(
                            (<[u8; NUM_CHANNELS]>::try_from(colors.next().unwrap())).unwrap(),
                        )
                    })
                });

            // Output `line` to `out` while assigning colors to each cell
            for (chr, palette) in line.chars().zip(line_palettes) {
                out.write_char_cell(chr, palette)?;
            }

            line.clear();
            out.write_line_terminator()?;
        }
        Ok(())
    }
}

struct ImageReadImpl<'a> {
    indices: ArrayView2<'a, bool>,
}

impl img2text::ImageRead for ImageReadImpl<'_> {
    #[inline]
    fn dims(&self) -> [usize; 2] {
        let (h, w) = self.indices.dim();
        [w, h] // `img2text` uses the `[width, height]` format
    }

    #[inline]
    fn copy_line_as_spans_to(&self, y: usize, out: &mut [img2text::Span]) {
        let row = self.indices.row(y);
        img2text::set_spans_by_fn(out, row.len(), |x| row[x]);
    }
}

/// A trait for writing pseudographical character cells.
pub trait WriteCell {
    type Error;

    /// Write a line terminator. This will be called for
    /// [`QuantizedImage::image_dim`]`[1]` times.
    fn write_line_terminator(&mut self) -> Result<(), Self::Error> {
        self.write_str_cell("\n", Default::default())
    }

    fn write_str_cell(&mut self, s: &str, palette: [rgb::RGB8; 2]) -> Result<(), Self::Error>;

    #[inline]
    fn write_char_cell(
        &mut self,
        c: char,
        [bg_color, fg_color]: [rgb::RGB8; 2],
    ) -> Result<(), Self::Error> {
        self.write_str_cell(c.encode_utf8(&mut [0; 4]), [bg_color, fg_color])
    }
}

/// A [`WriteCell`] implementation that outputs texts decorated by [24-bit color
/// ANSI escape sequences][1] to `W: `[`fmt::Write`][2].
///
/// [1]: https://en.wikipedia.org/w/index.php?title=ANSI_escape_code&oldid=1093948260#24-bit
/// [2]: core::fmt::Write
pub struct AnsiTrueColorCellWriter<W> {
    inner: W,
}

/// Control Sequence Introducer.
const CSI: &str = "\x1b[";

impl<W: fmt::Write> AnsiTrueColorCellWriter<W> {
    /// Construct a `Self`.
    #[inline]
    pub fn new(inner: W) -> Self {
        Self { inner }
    }
}

impl<W: fmt::Write> WriteCell for AnsiTrueColorCellWriter<W> {
    type Error = fmt::Error;

    #[inline]
    fn write_str_cell(
        &mut self,
        s: &str,
        [rgb::RGB8 {
            r: bg_r,
            g: bg_g,
            b: bg_b,
        }, rgb::RGB8 {
            r: fg_r,
            g: fg_g,
            b: fg_b,
        }]: [rgb::RGB8; 2],
    ) -> Result<(), Self::Error> {
        write!(
            self.inner,
            "{CSI}38;2;{fg_r};{fg_g};{fg_b}m\
            {CSI}48;2;{bg_r};{bg_g};{bg_b}m\
            {s}"
        )
    }

    #[inline]
    fn write_line_terminator(&mut self) -> Result<(), Self::Error> {
        // SGR 0 (reset attribute) + LR
        self.inner.write_str(CSI)?;
        self.inner.write_str("m\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn write_2x3(mut colors: Vec<u8>, indices: Vec<bool>, img_width: usize) -> TestResult {
        if img_width == 0 || img_width > colors.len() {
            return TestResult::discard();
        }
        let img_height = colors.len() / (img_width * NUM_CHANNELS * PALETTE_LEN);
        if img_height == 0 {
            return TestResult::discard();
        }
        colors.resize_with(
            PALETTE_LEN * NUM_CHANNELS * img_width * img_height,
            || unreachable!(),
        );

        let cell_dims = [3, 2];
        let mut image = QuantizedImage {
            indices: Array2::default((img_height * cell_dims[0], img_width * cell_dims[1])),
            palettes: Array4::from_shape_vec(
                (PALETTE_LEN, NUM_CHANNELS, img_height, img_width),
                colors,
            )
            .unwrap(),
        };
        indices
            .iter()
            .cycle()
            .zip(image.indices.iter_mut())
            .for_each(|(&index, out_index)| *out_index = index);
        image
            .write_2x3_to(&mut AnsiTrueColorCellWriter::new(String::new()))
            .unwrap();
        TestResult::passed()
    }
}
