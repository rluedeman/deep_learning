# A set of tools to efficiently build large, GAN-friendly image datasets
GAN-friendly = At least 20k images selected from a fairly narrow distribution.

### The tools
1. ImgScraper
2. ImgFilter
3. Trainer

## ImgScraper
A utility to scrape images from a variety of publicly available sources. It can be configured with various search filters to narrow the type of images it scrapes.

## ImgFilter
A utility that can be primed with human-curated images and then set loose on very large datasets to filter them down to similar images to the human-curated set.

## Trainer
A utility to manage the training pipeline for an image GAN.


# How to use them
1. start_gan_project: --name Food --flickr_search_terms="food on a plate" --num_starter_images=1000
2. scripts/calibrate_
