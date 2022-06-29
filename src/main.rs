use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueHint};
use std::{convert::TryInto, io::prelude::*, path::PathBuf, str::FromStr};

#[derive(Parser, Debug)]
#[clap(long_about = r"
Color image-to-text converter
")]
struct Opts {
    /// The image to process
    #[clap(name = "FILE", value_hint = ValueHint::AnyPath)]
    image_path: PathBuf,
    /// The glyph set to use
    #[clap(short = 'g', default_value = "2x3", value_enum)]
    style: Style,
    /// The output format
    #[clap(short = 'f', default_value = "ansi24", value_enum)]
    output_format: OutputFormat,
    /// The width of output characters, only used when `-s` is given without
    /// `!`
    #[clap(short = 'w', default_value = "0.45")]
    cell_width: f64,
    /// The output size, measured in character cells or percent (e.g., `80`,
    /// `80x40`, `80x40!`, `-80x40`, `100%`).
    /// [default: downscale to terminal size (if the output is a terminal) or
    /// 100% (otherwise)]
    ///
    ///  - 80: Fit within 80x80 character cells
    ///
    ///  - 80x40: Fit within 80x40 character cells, upscaling as necessary
    ///
    ///  - -80x40: Fit within 80x40 character cells, only downscaling
    ///
    ///  - 80x40!: Fit to 80x40 character cells, not maintaining the aspect
    ///    ratio
    ///
    ///  - 150%: Scale by 150%. The actual output size depends on the glyph set
    ///    being used; for example, `2x3` maps each 2x3 block to one character.
    ///
    #[clap(short = 's')]
    out_size: Option<SizeSpec>,

    /// Dithering strength in range `0..1`.
    #[clap(short = 'd', default_value = "1")]
    dither: f32,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum Style {
    _1x2,
    _2x2,
    _2x3,
    Slc,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum OutputFormat {
    Ansi24,
}

#[derive(Debug)]
enum SizeSpec {
    Absolute { dims: [usize; 2], mode: SizeMode },
    Relative(f64),
}

#[derive(Debug, PartialEq)]
enum SizeMode {
    Contain,
    Fill,
    ScaleDown,
}

impl FromStr for SizeSpec {
    type Err = String;

    fn from_str(mut s: &str) -> Result<Self, Self::Err> {
        if let Some(rest) = s.strip_suffix("%") {
            let ratio: f64 = rest.parse().map_err(|_| format!("bad ratio: '{}'", rest))?;

            if !ratio.is_finite() || ratio < 0.0 {
                return Err(format!("ratio out of range: '{}'", rest));
            }

            return Ok(Self::Relative(ratio / 100.0));
        }

        let force = if let Some(rest) = s.strip_suffix("!") {
            s = rest;
            true
        } else {
            false
        };

        let scale_down = if let Some(rest) = s.strip_prefix("-") {
            s = rest;
            true
        } else {
            false
        };

        let dims = if let Some(i) = s.find("x") {
            // width x height
            let width = &s[0..i];
            let height = &s[i + 1..];
            [
                width
                    .parse()
                    .map_err(|_| format!("bad width: '{}'", width))?,
                height
                    .parse()
                    .map_err(|_| format!("bad height: '{}'", height))?,
            ]
        } else {
            // size
            let size = s.parse().map_err(|_| format!("bad size: '{}'", s))?;
            [size, size]
        };

        Ok(Self::Absolute {
            dims,
            mode: match (force, scale_down) {
                (true, false) => SizeMode::Fill,
                (false, true) => SizeMode::ScaleDown,
                (false, false) => SizeMode::Contain,
                (true, true) => return Err("cannot specify both `!` and `-`".to_owned()),
            },
        })
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("img2ctext=info"))
        .init();

    let mut opts = Opts::parse();
    log::debug!("opts = {:#?}", opts);

    // Open the image
    let img = image::open(&opts.image_path).with_context(|| {
        format!(
            "Failed to read an input image from '{}'",
            opts.image_path.display()
        )
    })?;
    let mut img = img.into_rgb8();

    // Options
    let mut qopts = img2ctext::DEFAULT_QUANTIZE_OPTS;

    qopts.cell_dim = match opts.style {
        Style::_1x2 => [2, 1],
        Style::_2x2 => [2, 2],
        Style::_2x3 => [3, 2],
        Style::Slc => [3, 3],
    };
    qopts.dither_strength = opts.dither;

    if !opts.cell_width.is_finite() || opts.cell_width <= 0.1 || opts.cell_width > 10.0 {
        bail!("cell_width is out of range");
    }

    // Resize the image to the terminal size if the size is not specified
    let console_stdout = console::Term::stdout();
    if opts.out_size.is_none() && console_stdout.features().is_attended() {
        if let Some((h, w)) = console_stdout.size_checked() {
            let h = h.saturating_sub(3);
            log::info!(
                "downscaling to `{}x{}` (tty size minus some) because stdout is tty, and `-s` is unspecified",
                w,
                h
            );
            opts.out_size = Some(SizeSpec::Absolute {
                mode: SizeMode::ScaleDown,
                dims: [h as _, w as _],
            });
        }
    }

    // Resize the image if requested
    if let Some(out_size) = &opts.out_size {
        let in_dims = match out_size {
            SizeSpec::Absolute {
                dims,
                mode: SizeMode::Fill,
            } => adjust_image_size_for_output_size(*dims, qopts.cell_dim)
                .ok_or_else(|| anyhow!("requested size is too large"))?,

            SizeSpec::Absolute {
                dims,
                mode: SizeMode::Contain,
            } => adjust_image_size_for_output_size_preserving_aspect_ratio(
                [img.height() as _, img.width() as _],
                *dims,
                true,
                false, // contain
                opts.cell_width,
                qopts.cell_dim,
            )
            .ok_or_else(|| anyhow!("requested size is too large"))?,

            SizeSpec::Absolute {
                dims,
                mode: SizeMode::ScaleDown,
            } => adjust_image_size_for_output_size_preserving_aspect_ratio(
                [img.height() as _, img.width() as _],
                *dims,
                false,
                false, // contain
                opts.cell_width,
                qopts.cell_dim,
            )
            .ok_or_else(|| anyhow!("requested size is too large"))?,

            SizeSpec::Relative(ratio) => {
                let w = img.width() as f64 * ratio;
                let h = img.height() as f64 * ratio;
                if w > u32::MAX as f64 || h > u32::MAX as f64 {
                    bail!("requested size is too large");
                }
                [h as usize, w as usize]
            }
        };

        let in_dims = [in_dims[0] as u32, in_dims[1] as u32];

        if img.dimensions() != (in_dims[1], in_dims[0]) {
            log::debug!(
                "resampling the image from {:?} to {:?}",
                match img.dimensions() {
                    (x, y) => [x, y],
                },
                in_dims
            );

            img = image::imageops::resize(
                &img,
                in_dims[1],
                in_dims[0],
                image::imageops::FilterType::CatmullRom,
            );
        } else {
            log::debug!(
                "refusing to resample the image to the identical size ({:?})",
                in_dims
            );
        }
    }

    // Convert the image
    let (img_width, img_height) = img.dimensions();
    let img_linear = ndarray::Array3::from_shape_fn(
        (3, img_height as usize, img_width as usize),
        |(ch_i, y, x)| {
            let pixel = img.get_pixel(x as u32, y as u32);
            srgb::gamma::expand_u8(pixel[ch_i])
        },
    );

    // Quantize the image
    log::debug!("qopts = {qopts:?}");
    log::debug!("img_linear.dim = {:?}", img_linear.view().dim());
    let qimg = img2ctext::quantize(img_linear.view(), &qopts).unwrap(); // TODO: report errors

    log::debug!("qimg.palettes.dim = {:?}", qimg.palettes.dim());
    log::debug!("qimg.indices.dim = {:?}", qimg.indices.dim());
    log::debug!("qimg.cell_dim = {:?}", qimg.cell_dim());
    log::debug!("qimg.image_dim = {:?}", qimg.image_dim());

    // Output the image
    let output = fn_formats::DisplayFmt(|f: &mut _| {
        let mut writer = match opts.output_format {
            OutputFormat::Ansi24 => img2ctext::AnsiTrueColorCellWriter::new(f),
        };

        match opts.style {
            Style::_1x2 => qimg.write_1x2_to(&mut writer),
            Style::_2x2 => qimg.write_2x2_to(&mut writer),
            Style::_2x3 => qimg.write_2x3_to(&mut writer),
            Style::Slc => qimg.write_slc_to(&mut writer),
        }
    });

    std::io::stdout()
        .write_fmt(format_args!("{output}"))
        .context("Failed to write the output to the standard output")?;

    Ok(())
}

fn adjust_image_size_for_output_size_preserving_aspect_ratio(
    image_dims: [usize; 2],
    output_dims: [usize; 2],
    can_scale_up: bool,
    cover: bool,
    cell_width: f64,
    cell_dims: [usize; 2],
) -> Option<[usize; 2]> {
    // Calculate the "natural" size
    let [nat_out_w, nat_out_h] = [
        image_dims[0] as f64 / cell_dims[0] as f64,
        image_dims[1] as f64 / cell_dims[1] as f64,
    ];
    let aspect = cell_dims[1] as f64 / cell_dims[0] as f64 / cell_width;

    let [img_w, img_h] = [nat_out_w / aspect.max(1.0), nat_out_h * aspect.min(1.0)];
    log::debug!("'natural' output size = {:?}", [img_w, img_h]);
    let scale_x = output_dims[0] as f64 / img_w;
    let scale_y = output_dims[1] as f64 / img_h;

    let mut scale = if cover {
        f64::max(scale_x, scale_y)
    } else {
        f64::min(scale_x, scale_y)
    };
    if !can_scale_up {
        scale = scale.min(1.0);
    }
    log::debug!("scaling the 'natural' output size by {}...", scale);

    let output_dims = [
        (img_w * scale).round() as usize,
        (img_h * scale).round() as usize,
    ];

    adjust_image_size_for_output_size(output_dims, cell_dims)
}

fn adjust_image_size_for_output_size(
    output_dims: [usize; 2],
    cell_dims: [usize; 2],
) -> Option<[usize; 2]> {
    // FIXME: Use `zip` when stable
    Some([
        output_dims[0].checked_mul(cell_dims[0])?.try_into().ok()?,
        output_dims[1].checked_mul(cell_dims[1])?.try_into().ok()?,
    ])
}
struct GrayImageRead<'a> {
    image: &'a image::GrayImage,
    threshold: usize,
    invert: bool,
}

impl img2text::ImageRead for GrayImageRead<'_> {
    fn dims(&self) -> [usize; 2] {
        let (w, h) = self.image.dimensions();
        [w.try_into().unwrap(), h.try_into().unwrap()]
    }

    fn copy_line_as_spans_to(&self, y: usize, out: &mut [img2text::Span]) {
        let Self {
            image,
            threshold,
            invert,
        } = *self;
        img2text::set_spans_by_fn(out, self.dims()[0], move |x| {
            (image[(x as u32, y as u32)].0[0] as usize >= threshold) ^ invert
        });
    }
}

struct BlackWhiteColorMap {
    threshold: u8,
    palette: [u8; 2],
}

impl image::imageops::colorops::ColorMap for BlackWhiteColorMap {
    type Color = image::Luma<u8>;

    #[inline]
    fn index_of(&self, color: &Self::Color) -> usize {
        (color.0[0] >= self.threshold) as usize
    }

    #[inline]
    fn map_color(&self, color: &mut Self::Color) {
        *color = self.lookup(self.index_of(color)).unwrap();
    }

    #[inline]
    fn lookup(&self, index: usize) -> Option<Self::Color> {
        Some(image::Luma([self.palette[index]]))
    }

    fn has_lookup(&self) -> bool {
        true
    }
}
