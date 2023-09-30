import os
import argparse
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch_fidelity import calculate_metrics


def has_cache(cache_root, cache_name):
    return os.path.exists(os.path.join(cache_root, f'{cache_name}-inception-v3-compat-features-2048.pt')) \
        and os.path.exists(os.path.join(cache_root, f'{cache_name}-inception-v3-compat-features-logits_unbiased.pt')) \
        and os.path.exists(os.path.join(cache_root, f'{cache_name}-inception-v3-compat-stat-fid-2048.pt'))


def clear_cache(cache_root, cache_name):
    if os.path.exists(os.path.join(cache_root, f'{cache_name}-inception-v3-compat-features-2048.pt')):
        os.remove(os.path.join(cache_root, f'{cache_name}-inception-v3-compat-features-2048.pt'))
    if os.path.exists(os.path.join(cache_root, f'{cache_name}-inception-v3-compat-features-logits_unbiased.pt')):
        os.remove(os.path.join(cache_root, f'{cache_name}-inception-v3-compat-features-logits_unbiased.pt'))
    if os.path.exists(os.path.join(cache_root, f'{cache_name}-inception-v3-compat-stat-fid-2048.pt')):
        os.remove(os.path.join(cache_root, f'{cache_name}-inception-v3-compat-stat-fid-2048.pt'))


def confirm(prompt='Continue? (y/n)', action_if_yes=None, action_if_no=None):
    while True:
        print(prompt)
        choice = input()
        if choice == 'y':
            if action_if_yes is not None:
                action_if_yes()
            break
        elif choice == 'n':
            if action_if_no is not None:
                action_if_no()
            break


def center_crop_and_resize(image, image_size):
    w, h = image.size
    if w > h:
        image = image.crop(((w - h) // 2, 0, (w + h) // 2, h))
    elif h > w:
        image = image.crop((0, (h - w) // 2, w, (h + w) // 2))
    return image.resize((image_size, image_size), Image.LANCZOS)


def sample_from_fake_image(fake_images_dir, tmp_dir, image_size, num_samples=2000):
    print('Sampling fake images... Saving to', tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    fake_images = glob(os.path.join(fake_images_dir, '*.png'))
    print('Found', len(fake_images), 'fake images')
    fake_images = np.random.choice(fake_images, len(fake_images), replace=False)
    count = 0
    with tqdm(total=num_samples, desc='Sampling fake images') as pbar:
        for i, fake_image in enumerate(fake_images):
            try:
                fake_image = Image.open(fake_image)
                if i == 0:
                    assert fake_image.size == (image_size, image_size)
                fake_image.save(os.path.join(tmp_dir, f'{i:05d}.png'))
                count += 1
                pbar.update(1)
                if count == num_samples:
                    break
            except Exception as e:
                print(e)
                continue
    print('Saved', count, 'fake images')


def sample_from_real_image(real_images_dir, tmp_dir, image_size, num_samples=None):
    print('Sampling real images... Saving to', tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    real_images = []
    for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
        real_images.extend(glob(os.path.join(real_images_dir, '**', f'*.{ext}'), recursive=True))
    if num_samples is not None:
        real_images = np.random.choice(real_images, num_samples, replace=False)
    for i, real_image in enumerate(tqdm(real_images, desc='Sampling real images')):
        try:
            real_image = Image.open(real_image)
            if real_image.mode == "CMYK":
                real_image = real_image.convert("RGB")
            real_image = center_crop_and_resize(real_image, image_size)
            real_image.save(os.path.join(tmp_dir, f'{i:05d}.png'))
        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_images_dir', type=str, default=None, help='Real images directory')
    parser.add_argument('--fake_images_dir', type=str, default=None, help='Fake images directory')
    parser.add_argument('--tmp_dir', type=str, default='metrics/cache', help='Temporary directory')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--real_images_cache_name', type=str, default=None, help='Real images cache name')
    parser.add_argument('--fake_images_cache_name', type=str, default=None, help='Fake images cache name')
    parser.add_argument('--use_real_images_cache', action='store_true', help='Use real images cache')
    parser.add_argument('--use_fake_images_cache', action='store_true', help='Use fake images cache')
    opt = parser.parse_args()

    opt.real_images_cache_name = opt.real_images_dir.replace('/', '_') if opt.real_images_cache_name is None else opt.real_images_cache_name
    opt.fake_images_cache_name = opt.fake_images_dir.replace('/', '_') if opt.fake_images_cache_name is None else opt.fake_images_cache_name
    
    tmp_real_dir = os.path.join(opt.tmp_dir, opt.real_images_cache_name)
    tmp_fake_dir = os.path.join(opt.tmp_dir, opt.fake_images_cache_name)

    if opt.use_real_images_cache:
        if not has_cache(opt.tmp_dir, opt.real_images_cache_name):
            raise ValueError('Real images cache not found')
        print('Using cached real images')
    else:
        if has_cache(opt.tmp_dir, opt.real_images_cache_name):
            confirm('Real images cache found. Overwrite? (y/n)',
                    action_if_yes=lambda: clear_cache(opt.tmp_dir, opt.real_images_cache_name))
        if not has_cache(opt.tmp_dir, opt.real_images_cache_name):
            sample_from_real_image(opt.real_images_dir, tmp_real_dir, opt.image_size, None)
    if opt.use_fake_images_cache:
        if not has_cache(opt.tmp_dir, opt.fake_images_cache_name):
            raise ValueError('Fake images cache not found')
        print('Using cached fake images')
    else:
        if has_cache(opt.tmp_dir, opt.fake_images_cache_name):
            confirm('Fake images cache found. Overwrite? (y/n)',
                    action_if_yes=lambda: clear_cache(opt.tmp_dir, opt.fake_images_cache_name))
        if not has_cache(opt.tmp_dir, opt.fake_images_cache_name):
            sample_from_fake_image(opt.fake_images_dir, tmp_fake_dir, opt.image_size, opt.num_samples)

    metrics_dict = calculate_metrics(input1=tmp_fake_dir, input2=tmp_real_dir, cuda=True, isc=True, fid=True, kid=True, verbose=True,
        input1_cache_name=opt.fake_images_cache_name,
        input2_cache_name=opt.real_images_cache_name,
        cache_root=opt.tmp_dir
    )

    print(metrics_dict)
    with open(os.path.join(os.path.dirname(__file__), 'metrics', f'{opt.fake_images_cache_name}.txt'), 'w') as f:
        f.write(str(metrics_dict))
