//! Image quantization - the best part of this crate
use ndarray::{Array2, Array3, Array4, ArrayView3, Axis, Zip};

use crate::{image::QuantizedImage, NUM_CHANNELS, PALETTE_LEN};

/// Options for [`quantize`].
#[non_exhaustive]
#[derive(Debug)]
pub struct QuantizeOpts<'a> {
    /// The dithering amount. The standard range is `0..1`.
    pub dither_strength: f32,
    /// The cell size. The default value is `[3, 2]`, which is appropriate for
    /// use with [`QuantizedImage::write_2x3_to`].
    ///
    /// The resulting image's [`QuantizedImage::cell_dim`] will reflect this
    /// value.
    pub cell_dim: [usize; 2],
    /// Phantom covariant lifetime marker
    _unused: &'a (),
}

/// A `QuantizeOpts` with sensible default values.
pub const DEFAULT_QUANTIZE_OPTS: QuantizeOpts<'static> = QuantizeOpts {
    dither_strength: 1.0,
    cell_dim: [3, 2],
    _unused: &(),
};

impl<'a> Default for QuantizeOpts<'a> {
    #[inline]
    fn default() -> Self {
        DEFAULT_QUANTIZE_OPTS
    }
}

/// The error type for [`quantize`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum QuantizeError {
    /// One or both sides of the input image are smaller than `cell_dim`.
    ///
    /// `quantize` refuses to produce an empty [`QuantizedImage`], which would
    /// violate its invariant `[ref:qimage_non_empty]`.
    #[error("image too small")]
    ImageTooSmall,
    /// The number of input channels (`image.dim().0`) is not equal to
    /// [`NUM_CHANNELS`].
    #[error("unsupported channel count")]
    IncorrectChannelCount,
    /// `cell_dim` contains a zero element.
    #[error("invalid cell size")]
    EmptyCell,
    /// `cell_dim` exceeds an internal limit.
    ///
    /// `[tag:quantize_cell_limit]` The current limit is `cell_dim[0] *
    /// cell_dim[1] <= 32`.
    #[error("cell dimensions are too large")]
    TooLargeCell,
}

/// Construct a [`QuantizedImage`] from a `f32` normalized (`0..1`) linear
/// colorspace RGB image represented by an `ArrayView3` of size `[NUM_CHANNELS,
/// height, width]`.
pub fn quantize(
    image: ArrayView3<'_, f32>,
    &QuantizeOpts {
        dither_strength,
        cell_dim,
        _unused,
    }: &QuantizeOpts<'_>,
) -> Result<QuantizedImage, QuantizeError> {
    let image_dim = image.dim();
    if image_dim.0 != NUM_CHANNELS {
        return Err(QuantizeError::IncorrectChannelCount);
    }
    let image_dim = [image_dim.1, image_dim.2];
    let qimage_dim = [
        // FIXME: Use `array::try_map` when it's stable
        image_dim[0]
            .checked_div(cell_dim[0])
            .ok_or(QuantizeError::EmptyCell)?,
        image_dim[1]
            .checked_div(cell_dim[1])
            .ok_or(QuantizeError::EmptyCell)?,
    ];
    if qimage_dim[0] == 0 || qimage_dim[1] == 0 {
        return Err(QuantizeError::ImageTooSmall);
    }

    // Rounded image size
    // FIXME: Use `array::zip` when it's stable
    let image_dim = [qimage_dim[0] * cell_dim[0], qimage_dim[1] * cell_dim[1]];

    // [ref:quantize_cell_limit]
    let cell_area = cell_dim[0].saturating_mul(cell_dim[1]);
    if cell_area > 32 {
        return Err(QuantizeError::TooLargeCell);
    }
    let permutation_upper = 1u32 << (cell_area - 1);

    // A temporary array to hold colors in a subimage
    let mut subimage = Array3::zeros((NUM_CHANNELS, cell_dim[0], cell_dim[1]));

    // Initial color selection by exhaustive search
    // TODO: Try cluster fit [ref:ye2014]
    let mut palettes = Array4::zeros((PALETTE_LEN, NUM_CHANNELS, qimage_dim[0], qimage_dim[1]));
    Zip::from(image.exact_chunks((NUM_CHANNELS, cell_dim[0], cell_dim[1])))
        .and(
            palettes
                .view_mut()
                .into_shape((PALETTE_LEN * NUM_CHANNELS, qimage_dim[0], qimage_dim[1]))
                .unwrap()
                .exact_chunks_mut((PALETTE_LEN * NUM_CHANNELS, 1, 1)),
        )
        .for_each(|subimage_view, mut out_palette| {
            // Copy `subimage_view` to `subimage`. This may speed up the
            // process because we don't care about pixel order here, and
            // `subimage[i, .., ..]` is laid out contiguously on memory.
            subimage.assign(&subimage_view);
            let subimage = subimage
                .view_mut()
                .into_shape((NUM_CHANNELS, cell_area))
                .unwrap();

            // The bit `i` of `permistation` indicates which palette entry the
            // `i`-th input pixel from `subimage` should belong to.
            // Avoid all-zero (0) or all-one permutations (`permutation_upper
            // * 2 - 1`) because that would be pointless. We don't try either
            // the second half of all possible permutations (`permutation_upper
            // ..permutation_upper * 2`) because that's just an inverted version
            // of the first half.
            let (_, palette) = (1..permutation_upper)
                .map(|permutation| {
                    let color_index_at = |i| ((permutation >> i) & 1) as usize;
                    // Collect samples in two bins
                    let mut color_accumulator = [[0f32; NUM_CHANNELS]; PALETTE_LEN];
                    let num_samples = [
                        cell_area as u32 - permutation.count_ones(),
                        permutation.count_ones(),
                    ];
                    debug_assert_ne!(num_samples[0], 0);
                    debug_assert_ne!(num_samples[1], 0);

                    for (i, pixel) in subimage.columns().into_iter().enumerate() {
                        let color_accumulator = &mut color_accumulator[color_index_at(i)];
                        // FIXME: Use `array::zip` when stable
                        for (channel_accumulator, &pixel_channel) in
                            color_accumulator.iter_mut().zip(pixel.iter())
                        {
                            *channel_accumulator += pixel_channel;
                        }
                    }

                    // Calculate the mean for each bin. The result will be used as
                    // the palette for this subimage.
                    // FIXME: Use `array::zip` when stable
                    let mut palette = color_accumulator;
                    for (color, num_samples) in palette.iter_mut().zip(num_samples) {
                        let scale = 1.0f32 / num_samples as f32;
                        for channel in color.iter_mut() {
                            *channel *= scale;
                        }
                    }

                    // Evaluate this solution
                    let cost = subimage
                        .columns()
                        .into_iter()
                        .enumerate()
                        .map(|(i, pixel)| {
                            let palette_color = palette[color_index_at(i)];
                            palette_color
                                .iter()
                                .zip(pixel.iter())
                                .map(|(&c0, &c1)| (c0 - c1).powi(2))
                                .sum::<f32>()
                        })
                        .sum::<f32>();
                    (cost, palette)
                })
                // Find the best solution
                // n.b. Positive finite `f32` values are ordered by their
                // integral representations. `u32` comparison is also faster
                // on hardware in general.
                .min_by_key(|&(cost, _)| cost.to_bits())
                .unwrap();

            // Assign the result to `out_palette`
            out_palette
                .iter_mut()
                .zip(palette.iter().flatten())
                .for_each(|(out_chan, &chan)| *out_chan = chan);
        });
    drop(subimage);

    // Encode `palettes` by sRGB
    let palettes_srgb = palettes.map_mut(|channel| {
        let channel_srgb = srgb::gamma::compress_u8(*channel);
        *channel = srgb::gamma::expand_u8(channel_srgb);
        channel_srgb
    });

    // Diffused errors for the current and next rows. Used for dithering by
    // error diffusion. Includes one-pixel padding on both edges.
    let mut diffuse_rows = vec![[0.0; NUM_CHANNELS]; (image_dim[1] + 2) * 2];
    let (mut diffuse_cur_row, mut diffuse_next_row) = diffuse_rows.split_at_mut(image_dim[1] + 2);

    // Quantize the image with dithering by error diffusion. `indices` will
    // store the resulting color indices.
    let mut indices = Array2::default((image_dim[0], image_dim[1]));
    // Iterate by cell rows
    for ((mut out_cell_row_indices, cell_row_image), row_palettes) in indices
        .axis_chunks_iter_mut(Axis(0), cell_dim[0])
        .zip(image.axis_chunks_iter(Axis(1), cell_dim[0]))
        .zip(palettes.axis_iter(Axis(2)))
    {
        assert_eq!(out_cell_row_indices.dim().0, cell_dim[0]);
        assert_eq!(cell_row_image.dim().1, cell_dim[0]);

        // Iterate by rows
        for (mut out_row_indices, row_image) in out_cell_row_indices
            .axis_iter_mut(Axis(0))
            .zip(cell_row_image.axis_iter(Axis(1)))
        {
            let out_row_indices = out_row_indices.as_slice_mut().unwrap();

            // This row's diffused errors from the previous rows
            let mut interrow_diffuse = diffuse_cur_row[1..].iter_mut();

            // This row's diffused errors from already-processed pixels in this row
            let mut intrarow_diffuse = [0.0; NUM_CHANNELS];

            // Iterate by cell columns
            for ((out_cell_col_indices, cell_col_image), palette) in out_row_indices
                .chunks_exact_mut(cell_dim[1])
                .zip(row_image.axis_chunks_iter(Axis(1), cell_dim[1]))
                .zip(row_palettes.axis_iter(Axis(2)))
            {
                assert_eq!(cell_col_image.dim().1, cell_dim[1]);

                // TODO: Use `array::from_fn` when it's stable
                let mut palette_iter = palette.iter();
                let palette = [(); PALETTE_LEN]
                    .map(|()| [(); NUM_CHANNELS].map(|()| *palette_iter.next().unwrap()));

                // Iterate by columns
                for ((out_index, image_color), interrow_diffuse) in out_cell_col_indices
                    .iter_mut()
                    .zip(cell_col_image.axis_iter(Axis(1)))
                    .zip(interrow_diffuse.by_ref())
                {
                    let mut image_color_iter = image_color.iter();
                    // TODO: Use `array::from_fn` when it's stable
                    let mut image_color =
                        [(); NUM_CHANNELS].map(|()| *image_color_iter.next().unwrap());

                    // Apply the diffused error
                    for ((image_color, &interrow_diffuse), &intrarow_diffuse) in image_color
                        .iter_mut()
                        .zip(interrow_diffuse.iter())
                        .zip(intrarow_diffuse.iter())
                    {
                        *image_color += interrow_diffuse + intrarow_diffuse;
                    }

                    // Evaluate the candidates
                    let scores = palette.map(|palette_color| {
                        image_color
                            .iter()
                            .zip(palette_color.iter())
                            .map(|(&c0, &c1)| (c0 - c1).powi(2))
                            .sum::<f32>()
                    });

                    // Who's the winner?
                    let best_color_index = scores[1] < scores[0];
                    let quantized = palette[best_color_index as usize];
                    *out_index = best_color_index;

                    // Quantization error
                    let mut error = image_color;
                    for (error, &quantized) in error.iter_mut().zip(quantized.iter()) {
                        *error -= quantized;
                        *error *= dither_strength;
                    }

                    // Diffuse the error to the next pixel using the Floyd and
                    // Steinberg kernel.
                    for (intrarow_diffuse, &error) in intrarow_diffuse.iter_mut().zip(error.iter())
                    {
                        *intrarow_diffuse = error * (7.0 / 16.0);
                    }

                    // Replace `interrow_diffuse` with `error`
                    *interrow_diffuse = error;
                }
            }

            // `diffuse_cur_row` now contains this row's errors. Diffuse them
            // to the next row using the Floyd and Steinberg kernel.
            let kernel = [3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0];
            for (out_diffuse_next_row, diffuse_window_cur_row) in diffuse_next_row[1..]
                .iter_mut()
                .zip(diffuse_cur_row.windows(3))
            {
                for (i, out_diffuse_next_row) in out_diffuse_next_row.iter_mut().enumerate() {
                    *out_diffuse_next_row = diffuse_window_cur_row[0][i] * kernel[2]
                        + diffuse_window_cur_row[1][i] * kernel[1]
                        + diffuse_window_cur_row[2][i] * kernel[0];
                }
            }

            std::mem::swap(&mut diffuse_cur_row, &mut diffuse_next_row);
        }
    }

    // TODO: Try palette refinement by iterative non-linear optimization

    Ok(QuantizedImage {
        indices,
        palettes: palettes_srgb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn quantize_image(mut colors: Vec<u8>, dither_strength: f32, img_width: usize) -> TestResult {
        if img_width == 0 || img_width > colors.len() {
            return TestResult::discard();
        }
        let img_height = colors.len() / (img_width * NUM_CHANNELS);
        if img_height == 0 {
            return TestResult::discard();
        }
        colors.resize_with(NUM_CHANNELS * img_width * img_height, || unreachable!());
        let image = Array3::from_shape_vec((NUM_CHANNELS, img_height, img_width), colors)
            .unwrap()
            .map(|&c| (c as f32) / 255.0);

        let mut opts = DEFAULT_QUANTIZE_OPTS;
        opts.dither_strength = dither_strength;

        let qimg = match quantize(image.view(), &opts) {
            Ok(qimg) => qimg,
            Err(QuantizeError::ImageTooSmall) => {
                return TestResult::discard();
            }
            Err(e) => panic!("`quantize` unexpectedly failed: {e:?}"),
        };

        assert_eq!(qimg.cell_dim(), opts.cell_dim);

        TestResult::passed()
    }

    // TODO: Test loss-less conversion
}
