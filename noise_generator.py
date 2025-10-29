import numpy as np
import scipy.ndimage as ndi

def remove_comps(img_bool, img_spike):
    """Remove connected components from `img_spike` that are not connected to vessels in `img_bool`."""

    img_lab, num_comp = ndi.label(img_spike)
    comps_to_keep = np.unique(img_bool*img_lab)[1:]
    
    mask = np.zeros(num_comp+1, dtype=bool)
    mask[comps_to_keep] = True

    img_spike_rem = mask[img_lab]

    return img_spike_rem

def remove_small_comps(img_bin, tam_threshold):
    """For a binary image, remove connected components smaller than `tam_threshold`.

    Parameters
    ----------
    img_bin
        Binary image.
    tam_threshold
        Size threshold for removing components.

    Returns
    -------
    img_bin_final
        Binary image with small components removed.
    """

    img_lab, num_comp = ndi.label(img_bin)
    tam_comp = ndi.sum(img_bin, img_lab, range(num_comp+1))

    mask = tam_comp>tam_threshold
    mask[0] = False

    img_bin_final = mask[img_lab]

    return img_bin_final

def create_spikes(img_bool, spike_p):
    """Create spikes in the image."""

    img_spike = img_bool.copy()

    spike_mask = np.random.rand(*img_bool.shape) < spike_p
    img_spike[~img_bool & spike_mask] = True
    img_spike_rem = remove_comps(img_bool, img_spike)

    return img_spike_rem

def create_fps(img_bool, fp_p, fp_comp_threshold):
    """Create false positives (islands) in the image."""

    img_fp = img_bool.copy()

    s = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    img_bool_dil = ndi.binary_dilation(img_bool, structure=s)

    fp_mask = np.random.rand(*img_bool.shape) < fp_p
    fp_mask = ~img_bool_dil & fp_mask
    fp_mask = remove_small_comps(fp_mask, fp_comp_threshold)

    img_fp[~img_bool & fp_mask] = True

    return img_fp

def create_fns(img_bool, fn_p, fn_comp_threshold):
    """Create false negatives (holes) in the image."""

    img_fn = img_bool.copy()

    fn_mask = np.random.rand(*img_bool.shape) < fn_p
    fn_mask = remove_small_comps(fn_mask, fn_comp_threshold)

    img_fn[fn_mask] = False

    return img_fn

def create_noisy_img(
        img: np.ndarray,
        spike_amount: float,
        fp_p: float,
        fp_comp_threshold: int,
        fn_p: float,
        fn_comp_threshold: int
        ) -> np.ndarray:
    """ Create a noisy image with spikes, false positives (islands), and false negatives (holes).
    
    Parameters
    ----------
    img
        Binary image
    spike_amount
        Amount of spikes to add to the image. Must be between 0 and 1.
    fp_p
        Probability of false positives (islands).
    fp_comp_threshold
        Threshold for false positives. Connected components smaller than this value are removed.
    fn_p
        Probability of false negatives (holes).
    fn_comp_threshold
        Threshold for false negatives. Holes smaller than this value are removed.

    Returns
    -------
    img_spike_fp_fn
        Noisy image.
    """

    img_bool = img > 0

    img_spike = create_spikes(img_bool, spike_amount)
    img_spike_fp = create_fps(img_spike, fp_p, fp_comp_threshold)
    img_spike_fp_fn = create_fns(img_spike_fp, fn_p, fn_comp_threshold)

    return img_spike_fp_fn


