use anyhow::{anyhow, Result};
use image::{ImageBuffer, RgbImage};
use std::{fmt::Display, ops::Div, path::Path};
use venum::{conv::Mode, NaiveTensor};

fn main() -> Result<()> {
    let (image, width, height) = read_image("assets/venom.png")?;
    let image_tensor = NaiveTensor::new(&image, &[3, height as usize, width as usize])?;

    // Gaussian blur

    let blur_kernel = NaiveTensor::new(
        &[
            1.0, 2.0, 1.0, // Gaussian
            2.0, 4.0, 2.0, // Blur
            1.0, 2.0, 1.0, // Kernel
        ],
        &[3, 3],
    )?
    .div(16.0)?;

    let blur = image_tensor.correlate_2d(&blur_kernel, &[1, 1], Mode::Same)?;
    write_image(&blur, "assets/blur.png")?;

    // Sharpen

    let sharpen_kernel = NaiveTensor::new(
        &[
            0.0, -1.0, 0.0, // Sharpen
            -1.0, 5.0, -1.0, // Features
            0.0, -1.0, 0.0, // Kernel
        ],
        &[3, 3],
    )?;

    let sharpen = image_tensor.correlate_2d(&sharpen_kernel, &[1, 1], Mode::Same)?;
    write_image(&sharpen, "assets/sharpen.png")?;

    // Edge

    let edge_kernel = NaiveTensor::new(
        &[
            0.0, -1.0, 0.0, // Laplacian
            -1.0, 4.0, -1.0, // Edge Detection
            0.0, -1.0, 0.0, // Kernel
        ],
        &[3, 3],
    )?;

    let edge = image_tensor.correlate_2d(&edge_kernel, &[1, 1], Mode::Same)?;
    write_image(&edge, "assets/edge.png")?;

    let green_edge_kernel = NaiveTensor::new(
        &[
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            //
            0.0, -1.0, 0.0, // Laplacian
            -1.0, 4.0, -1.0, // Edge Detection
            0.0, -1.0, 0.0, // Kernel
            //
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
        ],
        &[3, 3, 3],
    )?;

    let green_edge = image_tensor.correlate_2d(&green_edge_kernel, &[1, 1], Mode::Same)?;
    write_image(&green_edge, "assets/green_edge.png")?;

    Ok(())
}

fn read_image<P>(path: P) -> Result<(Vec<f32>, u32, u32)>
where
    P: AsRef<Path>,
{
    let img = image::open(path)?.to_rgb8();
    let (width, height) = img.dimensions();

    let channel_size = (width * height) as usize;
    let channel_size_2 = channel_size * 2;
    let mut f32_data = vec![0.0; channel_size * 3];

    for (i, pixel) in img.pixels().enumerate() {
        f32_data[i] = pixel[0] as f32;
        f32_data[i + channel_size] = pixel[1] as f32;
        f32_data[i + channel_size_2] = pixel[2] as f32;
    }

    Ok((f32_data, width, height))
}

fn write_image<P>(tensor: &NaiveTensor<f32>, path: P) -> Result<()>
where
    P: AsRef<Path> + Display,
{
    let data = tensor.data();
    let sizes = tensor.sizes();
    let width = sizes[2] as u32;
    let height = sizes[1] as u32;

    let channel_size = (width * height) as usize;
    let channel_size_2 = channel_size * 2;
    let mut u8_data = Vec::with_capacity(channel_size * 3);

    for i in 0..channel_size {
        u8_data.push(data[i] as u8);
        u8_data.push(data[i + channel_size] as u8);
        u8_data.push(data[i + channel_size_2] as u8);
    }

    let img: RgbImage =
        ImageBuffer::from_raw(width, height, u8_data).ok_or(anyhow!("Error saving image."))?;

    img.save(&path)?;
    println!("Image saved at {}", path);

    Ok(())
}
