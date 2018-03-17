float convert_rgb_to_luma(float4 bgra, uint option)
{
  float luma;
  switch (option)
  {//https://stackoverflow.com/a/596243/7150241, with some modification from https://en.wikipedia.org/wiki/Luma_(video)
    case 1: //(1) CCIR 601 Luminance (perceived option 1)
      luma = sqrt(0.2126f * bgra.z + 0.7152f * bgra.y + 0.0722f * bgra.x);
      break;
    case 2: //(2) Luminance (perceived option 2, slower to calculate)
      luma = sqrt(0.299f * bgra.z * bgra.z + 0.587f * bgra.y * bgra.y + 0.114f * bgra.x * bgra.x);
      break;
    case 0: //(0) Relative Luminance (standard for certain colour spaces)
    default:
      luma = sqrt(0.2989f * bgra.z + 0.587f * bgra.y + 0.114f * bgra.x);
      break;
  }
  return luma;
}
__kernel void SobelEdgeDetection(__read_only image2d_t srcImg, __write_only image2d_t dstImg)
{
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
    CLK_ADDRESS_CLAMP_TO_EDGE | //Clamp to zeros
    CLK_FILTER_LINEAR;
  int2 coord = (int2)(get_global_id(0), get_global_id(1)); //coord.x, coord.y
  int max_width = get_image_width(srcImg);
  int max_height = get_image_height(srcImg);

  //row1
  uint4 bgra_m1m1 = read_imageui(srcImg, smp, (int2)(coord.x-1, coord.y-1));
  uint4 bgra_m1p0 = read_imageui(srcImg, smp, (int2)(coord.x-1, coord.y));
  uint4 bgra_m1p1 = read_imageui(srcImg, smp, (int2)(coord.x-1, coord.y+1));
  //row2
  uint4 bgra_p0m1 = read_imageui(srcImg, smp, (int2)(coord.x, coord.y-1));
  uint4 bgra_p0p0 = read_imageui(srcImg, smp, (int2)(coord.x, coord.y));
  uint4 bgra_p0p1 = read_imageui(srcImg, smp, (int2)(coord.x, coord.y+1));
  //row3
  uint4 bgra_p1m1 = read_imageui(srcImg, smp, (int2)(coord.x+1, coord.y-1));
  uint4 bgra_p1p0 = read_imageui(srcImg, smp, (int2)(coord.x+1, coord.y));
  uint4 bgra_p1p1 = read_imageui(srcImg, smp, (int2)(coord.x+1, coord.y+1));

  //row1
  float4 bgra_float_m1m1 = convert_float4(bgra_m1m1) / 255.0f;
  float4 bgra_float_m1p0 = convert_float4(bgra_m1p0) / 255.0f;
  float4 bgra_float_m1p1 = convert_float4(bgra_m1p1) / 255.0f;
  //row2
  float4 bgra_float_p0m1 = convert_float4(bgra_p0m1) / 255.0f;
  float4 bgra_float_p0p0 = convert_float4(bgra_p0p0) / 255.0f;
  float4 bgra_float_p0p1 = convert_float4(bgra_p0p1) / 255.0f;
  //row3
  float4 bgra_float_p1m1 = convert_float4(bgra_p1m1) / 255.0f;
  float4 bgra_float_p1p0 = convert_float4(bgra_p1p0) / 255.0f;
  float4 bgra_float_p1p1 = convert_float4(bgra_p1p1) / 255.0f;

  uint luma_option = 0; //change luminance choice for different formula used for converting rgb to grey
  //row1
  float luminance_m1m1 = convert_rgb_to_luma(bgra_float_m1m1, luma_option);
  float luminance_m1p0 = convert_rgb_to_luma(bgra_float_m1p0, luma_option);
  float luminance_m1p1 = convert_rgb_to_luma(bgra_float_m1p1, luma_option);
  //row2
  float luminance_p0m1 = convert_rgb_to_luma(bgra_float_p0m1, luma_option);
  float luminance_p0p0 = convert_rgb_to_luma(bgra_float_p0p0, luma_option);
  float luminance_p0p1 = convert_rgb_to_luma(bgra_float_p0p1, luma_option);
  //row3
  float luminance_p1m1 = convert_rgb_to_luma(bgra_float_p1m1, luma_option);
  float luminance_p1p0 = convert_rgb_to_luma(bgra_float_p1p0, luma_option);
  float luminance_p1p1 = convert_rgb_to_luma(bgra_float_p1p1, luma_option);

  //row1
  uint intensity_m1m1 = (uint) (luminance_m1m1 * 255.0f);
  uint intensity_m1p0 = (uint) (luminance_m1p0 * 255.0f);
  uint intensity_m1p1 = (uint) (luminance_m1p1 * 255.0f);
  //row2
  uint intensity_p0m1 = (uint) (luminance_p0m1 * 255.0f);
  uint intensity_p0p0 = (uint) (luminance_p0p0 * 255.0f);
  uint intensity_p0p1 = (uint) (luminance_p0p1 * 255.0f);
  //row3
  uint intensity_p1m1 = (uint) (luminance_p1m1 * 255.0f);
  uint intensity_p1p0 = (uint) (luminance_p1p0 * 255.0f);
  uint intensity_p1p1 = (uint) (luminance_p1p1 * 255.0f);

  uint gradient_x = intensity_m1m1 * (-1) + intensity_m1p1 * (+1)
                  + intensity_p0m1 * (-2) + intensity_p0p1 * (+2)
                  + intensity_p1m1 * (-1) + intensity_p1p1 * (+1);

  uint gradient_y = intensity_m1m1 * (-1) + intensity_m1p0 * (-2) + intensity_m1p1 * (-1)
                  + intensity_p1m1 * (+1) + intensity_p1p0 * (+2) + intensity_p1p1 * (+1);

  bgra_p0p0.x = bgra_p0p0.y = bgra_p0p0.z = (uint) (sqrt(gradient_x * gradient_x + gradient_y * gradient_y));
  bgra_p0p0.w = 255;
  write_imageui(dstImg, coord, bgra_p0p0);
}

__kernel void imagingTest(__read_only  image2d_t srcImg,
                       __write_only image2d_t dstImg)
{
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
    CLK_ADDRESS_CLAMP_TO_EDGE | //Clamp to zeros
    CLK_FILTER_LINEAR;
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  uint4 bgra = read_imageui(srcImg, smp, coord); //The byte order is BGRA
  float4 bgrafloat = convert_float4(bgra) / 255.0f; //Convert to normalized [0..1] float
  //Convert RGB to luminance (make the image grayscale).
  float luminance =  sqrt(0.241f * bgrafloat.z * bgrafloat.z + 0.691f * 
                      bgrafloat.y * bgrafloat.y + 0.068f * bgrafloat.x * bgrafloat.x);
  bgra.x = bgra.y = bgra.z = (uint) (luminance * 255.0f);
  bgra.w = 255;
  write_imageui(dstImg, coord, bgra);
}