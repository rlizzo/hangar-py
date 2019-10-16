from functools import partial
import warnings
from typing import Sequence
import random

from .common import GroupedAsets
from ..utils import LazyImporter

try:
    tf = LazyImporter('tensorflow')
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    raise ImportError(
        'Could not import "tensorflow" library. Ensure library is '
        'installed correctly to use tensorflow dataloader functions')


def yield_data(arraysets, sample_names, shuffle=False):
    if shuffle:
        sample_names = list(sample_names)
        random.shuffle(sample_names)
    for name in sample_names:
        yield tuple([aset[name] for aset in arraysets])


def make_tf_dataset(arraysets,
                    keys: Sequence[str] = None,
                    index_range: slice = None,
                    shuffle: bool = True):
    """
    Uses the hangar arraysets to make a tensorflow dataset. It uses
    `from_generator` function from `tensorflow.data.Dataset` with a generator
    function that wraps all the hangar arraysets. In such instances tensorflow
    Dataset does shuffle by loading the subset of data which can fit into the
    memory and shuffle that subset. Since it is not really a global shuffle
    `make_tf_dataset` accepts a `shuffle` argument which will be used by the
    generator to shuffle each time it is being called.

    .. warning::

        `tf.data.Dataset.from_generator` currently uses `tf.compat.v1.py_func()`
        internally. Hence the serialization function (`yield_data`) will not be
        serialized in a GraphDef. Therefore, you won't be able to serialize your
        model and restore it in a different environment if you use
        `make_tf_dataset`. The operation must run in the same address space as the
        Python program that calls tf.compat.v1.py_func(). If you are using
        distributed TensorFlow, you must run a tf.distribute.Server in the same
        process as the program that calls tf.compat.v1.py_func() and you must pin
        the created operation to a device in that server (e.g. using with
        tf.device():)

    Parameters
    ----------
    arraysets : :class:`~hangar.arrayset.ArraysetDataReader` or Sequence
        A arrayset object, a tuple of arrayset object or a list of arrayset
        objects`
    keys : Sequence[str]
        An iterable of sample names. If given only those samples will fetched from
        the arrayset
    index_range : slice
        A python slice object which will be used to find the subset of arrayset.
        Argument `keys` takes priority over `index_range` i.e. if both are given,
        keys will be used and `index_range` will be ignored
    shuffle : bool
        generator uses this to decide a global shuffle accross all the samples is
        required or not. But user doesn't have any restriction on
        doing`arrayset.shuffle()` on the returned arrayset

    Examples
    --------
    >>> from hangar import Repository
    >>> from hangar import make_tf_dataset
    >>> import tensorflow as tf
    >>> tf.compat.v1.enable_eager_execution()
    >>> repo = Repository('.')
    >>> co = repo.checkout()
    >>> data = co.arraysets['mnist_data']
    >>> target = co.arraysets['mnist_target']
    >>> tf_dset = make_tf_dataset([data, target])
    >>> tf_dset = tf_dset.batch(512)
    >>> for bdata, btarget in tf_dset:
    ...     print(bdata.shape, btarget.shape)


    Returns
    -------
    :class:`tf.data.Dataset`
    """
    warnings.warn("Dataloaders are experimental in the current release.", UserWarning)
    gasets = GroupedAsets(arraysets, keys, index_range)
    generator = partial(yield_data, gasets.arrayset_array, gasets.sample_names, shuffle)
    res = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=gasets.get_types(converter=tf.as_dtype),
        output_shapes=gasets.get_shapes(converter=tf.TensorShape))
    return res
