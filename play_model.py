import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

config.result_dir = 'play_results'

url_ffhq        = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()


def load_Gs(url):
    if url not in _Gs_cache:
        with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
            _G, _D, Gs = pickle.load(f)
        _Gs_cache[url] = Gs
    return _Gs_cache[url]


def create_transfer_gif_by_dlatents(Gs, src_dlatents, dst_dlatents, style_range, frame_num, target_file_name, save_src=False, save_dst=False):
    # interpolation
    dlatents = []
    for i in range(frame_num):
        cur_dlatent = np.zeros(src_dlatents.shape) # [1, 18, 512]
        cur_dlatent[...] = src_dlatents
        cur_dlatent[:, style_range] = src_dlatents[:, style_range] + (dst_dlatents[:, style_range] - src_dlatents[:, style_range]) * (i / (frame_num - 1))
        dlatents.append(cur_dlatent)

    # generate images
    dlatents = np.concatenate(dlatents) # [total_num, 18, 512]
    images = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs) # [total_num, 1024, 1024, 3]
    images = [PIL.Image.fromarray(images[i]) for i in range(images.shape[0])]

    if save_src:
        images[0].save(os.path.join(config.result_dir, 'src.png'))

    if save_dst:
        images[-1].save(os.path.join(config.result_dir, 'dst.png'))

    images[0].save(
        os.path.join(config.result_dir, target_file_name),
        format='GIF',
        append_images=images[1:],
        save_all=True,
        duration=150,
        loop=0)


def main():
    tflib.init_tf()
    os.makedirs(config.result_dir, exist_ok=True)

    Gs = load_Gs(url_ffhq)
    print(Gs.input_shape) # [None, 512]

    src_seed = 256
    dst_seed = 4096

    src_latents = np.random.RandomState(src_seed).randn(1, Gs.input_shape[1]) # [1, 512]
    dst_latents = np.random.RandomState(dst_seed).randn(1, Gs.input_shape[1]) # [1, 512]

    src_dlatents = Gs.components.mapping.run(src_latents, None) # [1, 18, 512]
    dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [1, 18, 512]

    # gif frame total number
    total_num = 20

    create_transfer_gif_by_dlatents(Gs, src_dlatents, dst_dlatents, range(0, 18), total_num, 'src_2_dst.gif', True, True)
    create_transfer_gif_by_dlatents(Gs, src_dlatents, dst_dlatents, range(0, 4), total_num, 'src_2_dst_0_4.gif')
    create_transfer_gif_by_dlatents(Gs, src_dlatents, dst_dlatents, range(4, 8), total_num, 'src_2_dst_4_8.gif')
    create_transfer_gif_by_dlatents(Gs, src_dlatents, dst_dlatents, range(8, 18), total_num, 'src_2_dst_8_18.gif')


if __name__ == '__main__':
    main()
