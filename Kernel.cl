__kernel void edgeDetection(__read_only image2d_t srcImg, __write_only image2d_t dstImg)
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

  //row1
  float luminance_m1m1 =  sqrt(0.241f * bgra_float_m1m1.z * bgra_float_m1m1.z + 0.691f * 
                    bgra_float_m1m1.y * bgra_float_m1m1.y + 0.068f * bgra_float_m1m1.x * bgra_float_m1m1.x);
  float luminance_m1p0 =  sqrt(0.241f * bgra_float_m1p0.z * bgra_float_m1p0.z + 0.691f * 
                    bgra_float_m1p0.y * bgra_float_m1p0.y + 0.068f * bgra_float_m1p0.x * bgra_float_m1p0.x);
  float luminance_m1p1 =  sqrt(0.241f * bgra_float_m1p1.z * bgra_float_m1p1.z + 0.691f * 
                    bgra_float_m1p1.y * bgra_float_m1p1.y + 0.068f * bgra_float_m1p1.x * bgra_float_m1p1.x);
  //row2
  float luminance_p0m1 =  sqrt(0.241f * bgra_float_p0m1.z * bgra_float_p0m1.z + 0.691f * 
                    bgra_float_p0m1.y * bgra_float_p0m1.y + 0.068f * bgra_float_p0m1.x * bgra_float_p0m1.x);
  float luminance_p0p0 =  sqrt(0.241f * bgra_float_p0p0.z * bgra_float_p0p0.z + 0.691f * 
                    bgra_float_p0p0.y * bgra_float_p0p0.y + 0.068f * bgra_float_p0p0.x * bgra_float_p0p0.x);
  float luminance_p0p1 =  sqrt(0.241f * bgra_float_p0p1.z * bgra_float_p0p1.z + 0.691f * 
                    bgra_float_p0p1.y * bgra_float_p0p1.y + 0.068f * bgra_float_p0p1.x * bgra_float_p0p1.x);
  //row3
  float luminance_p1m1 =  sqrt(0.241f * bgra_float_p1m1.z * bgra_float_p1m1.z + 0.691f * 
                    bgra_float_p1m1.y * bgra_float_p1m1.y + 0.068f * bgra_float_p1m1.x * bgra_float_p1m1.x);
  float luminance_p1p0 =  sqrt(0.241f * bgra_float_p1p0.z * bgra_float_p1p0.z + 0.691f * 
                    bgra_float_p1p0.y * bgra_float_p1p0.y + 0.068f * bgra_float_p1p0.x * bgra_float_p1p0.x);
  float luminance_p1p1 =  sqrt(0.241f * bgra_float_p1p1.z * bgra_float_p1p1.z + 0.691f * 
                    bgra_float_p1p1.y * bgra_float_p1p1.y + 0.068f * bgra_float_p1p1.x * bgra_float_p1p1.x);

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

  // uint4 bgra = read_imageui(srcImg, smp, (int2)(coord.x, coord.y));
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