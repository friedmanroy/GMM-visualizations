import numpy as np
from typing import Union
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
_imarray = Union[list, np.ndarray, tuple]


def _factors(num: int):
    return np.where((num % np.arange(1, np.floor(np.sqrt(num) + 1))) == 0)[0] + 1


def mosaic(images: _imarray, reshape: tuple=None, gap: int=1,
           normalize: bool=False, clip: bool=True, cols: int=-1,
           aspect_ratio: float=None) -> np.ndarray:
    """
    Create a mosaic of images
    :param images: a list or numpy array of images to mosaic. If "images" is a numpy array, then the first dimension
                   must be the number of images to mosaic
    :param reshape: if a tuple is given, each image will be reshaped according to the tuple. Default is None
    :param gap: the gap, in pixels, to add between each of the images. Default is 1px.
    :param normalize: if normalize is True, all of the images will be normalized to the scale between 0 and 1. In
                       other words, if normalize is True then for each image I<-(I-min(I))/(max(I)-min(I))
    :param clip: if clip is True, all the pixel values of each of the images will be clipped to the range [0, 1]
    :param cols: the number of columns the mosaic should have. If this is not given, then the number of columns will be
                 chosen automatically
    :param aspect_ratio: the aspect ratio of the resulting mosaic
    :return: a single numpy array that is the mosaic of all of the images
    """
    if cols > 0: assert len(images) % cols == 0, 'Bad number of columns given to mosaic'
    elif aspect_ratio is not None:
        cols = int(np.ceil(np.sqrt(len(images)*aspect_ratio)))
        while cols > 1 and not len(images) % cols:
            cols -= 1
    else: cols = len(images)//_factors(len(images))[-1]
    ims = images

    if normalize:
        ims = [(I-np.min(I))/(np.max(I)-np.min(I)) if np.max(I) != np.min(I) else I for I in ims]

    if clip:
        ims = [np.clip(I, 0, 1) for I in ims]

    if reshape is not None:
        ims = [I.reshape(reshape) for I in ims]

    if gap > 0:
        sh = (ims[0].shape[0], gap) if ims[0].ndim < 3 else (ims[0].shape[0], gap, 3)
        ims = [np.concatenate([np.ones(sh), I], axis=1) for I in ims]

    rows = [np.concatenate(ims[i*cols: (i+1)*cols], axis=1) for i in range(len(ims)//cols)]

    if gap > 0:
        sh = (gap, rows[0].shape[1]) if rows[0].ndim < 3 else (gap, rows[0].shape[1], 3)
        rows = [np.concatenate([np.ones(sh), I], axis=0) for I in rows]

    ret = np.concatenate(rows, axis=0)

    if gap > 0:
        sh = (gap, ret.shape[1]) if ims[0].ndim < 3 else (gap, ret.shape[1], 3)
        ret = np.concatenate([ret, np.ones(sh)], axis=0)
        sh = (ret.shape[0], gap) if ims[0].ndim < 3 else (ret.shape[0], gap, 3)
        ret = np.concatenate([ret, np.ones(sh)], axis=1)

    return ret


def _add_sep(mos: np.ndarray, sz: tuple, pad: int, sep_weight: int, cmax: int, rmax: int,
             sep_color: float=.3):
    if mos.ndim == 2:
        mos = np.concatenate([np.ones((cmax + pad, mos.shape[1])), mos], axis=0)
        mos = np.concatenate([np.ones((mos.shape[0], rmax + pad)), mos], axis=1)
    else:
        mos = np.concatenate([np.ones((cmax + pad, mos.shape[1], 3)), mos], axis=0)
        mos = np.concatenate([np.ones((mos.shape[0], rmax + pad, 3)), mos], axis=1)
    pad = (pad - sep_weight)//2
    if sep_weight > 0:
        for i in range(sep_weight):
            mos[cmax + pad + (2*pad + sep_weight) + i::sz[0] + 2*pad + sep_weight, :] = sep_color
            mos[:, rmax + pad + (2*pad + sep_weight) + i::sz[1] + 2*pad + sep_weight] = sep_color
    return mos


def _draw_titles(mos: np.ndarray, font: ImageFont, sz: tuple, pad: int, cmax: int, rmax: int,
                 col_titles: list, row_titles: list):
    im = Image.fromarray((mos*255).astype(np.uint8))
    draw = ImageDraw.Draw(im)

    col_start = 2*pad + rmax + sz[1]//2
    for i, c in enumerate(col_titles):
        w, h = draw.textsize(c, font)
        w, h = w/2, h/2
        x = col_start + (sz[1] + pad)*i
        y = (cmax + pad)//2
        draw.text((x-w, y-h), c, font=font, fill='black')

    row_start = 2*pad + cmax + sz[0]//2
    for i, r in enumerate(row_titles):
        w, h = draw.textsize(r, font)
        w, h = w//2, h//2
        y = row_start + (sz[0] + pad)*i
        x = (rmax + pad)//2
        draw.text((x - w, y - h), r, font=font, fill='black')
    return np.array(im).astype(np.float32)/255


def _parse_dict(im_dict: dict):
    keys = list(im_dict.keys())
    assert type(keys[0]) is tuple and len(keys[0]) == 2
    assert type(im_dict[keys[0]]) is np.ndarray

    blank = np.ones(im_dict[keys[0]].shape)
    im_dict = defaultdict(lambda: blank, im_dict)

    row_titles = [k[0] for k in keys]
    row_titles = list(np.array(row_titles)[np.sort(np.unique(row_titles, return_index=True)[1])])

    col_titles = [k[1] for k in keys]
    col_titles = list(np.array(col_titles)[np.sort(np.unique(col_titles, return_index=True)[1])])

    return [im_dict[(r, c)] for r in row_titles for c in col_titles], row_titles, col_titles


def ImageTable(images: Union[_imarray, dict], pad: float=.05,
               normalize: bool=False, clip: bool=True, col_titles: list=None, row_titles: list=None,
               fontsize: int=10, sep_weight: int=1,
               font_path: str='arial.ttf') -> np.ndarray:
    font = ImageFont.truetype(font_path, fontsize)

    if type(images) is dict:
        images, r_titles, c_titles = _parse_dict(images)
        if row_titles is None and col_titles is None:
            row_titles, col_titles = r_titles, c_titles

    sz = images[0].shape
    pad = int(np.ceil(min(images[0].shape[0], images[0].shape[1]) * pad))

    if sep_weight > 0:
        pad = sep_weight + 2*pad

    if col_titles is None and row_titles is None:
        cols = len(images)//_factors(len(images))[-1]
    elif col_titles is not None:
        cols = len(col_titles)
        if row_titles is None: row_titles = ['' for _ in range(len(images)//cols)]
    else:
        cols = len(images) // len(row_titles)
        col_titles = ['' for _ in range(cols)]

    if col_titles is not None:
        cmax = np.max([font.getsize(c)[1] for c in col_titles])
        rmax = np.max([font.getsize(r)[0] for r in row_titles])
    else:
        cmax = 0
        rmax = 0

    mos = mosaic(images, gap=pad, normalize=normalize, clip=clip, cols=cols)
    mos = _add_sep(mos, sz, pad, sep_weight, cmax, rmax)

    if col_titles is not None:
        mos = _draw_titles(mos, font, sz, pad, cmax, rmax, col_titles, row_titles)

    return mos
