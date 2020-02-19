"""Accessor column containing nested mapping of data under top level keys.

All backends are supported.
"""
from contextlib import ExitStack
from pathlib import Path
from typing import (
    Tuple, List, Union, Sequence, Dict, Iterable, Any, Type, Optional
)
from weakref import proxy

import numpy as np

from .common import open_file_handles
from ..records import (
    data_record_db_val_from_digest,
    data_record_digest_val_from_db_val,
    nested_data_db_key_from_names,
    hash_data_db_key_from_raw_key,
    schema_db_key_from_column,
    schema_hash_db_key_from_digest,
    schema_db_val_from_spec,
)
from ..records.parsing import generate_sample_name
from ..backends import backend_decoder, BACKEND_ACCESSOR_MAP, DataHashSpecsType
from ..op_state import reader_checkout_only
from ..utils import is_suitable_user_key, valfilter, valfilterfalse

AsetTxnType = Type['AsetTxn']
KeyType = Union[str, int]
EllipsisType = type(Ellipsis)
GetKeysType = Union[KeyType, EllipsisType, slice]
KeyArrMap = Dict[KeyType, np.ndarray]
KeyArrType = Union[Tuple[KeyType, np.ndarray], List[Union[KeyType, np.ndarray]]]
MapKeyArrType = Union[KeyArrMap, Sequence[KeyArrType]]


class FlatSubsampleReader(object):

    __slots__ = ('_column_name', '_stack', '_be_fs', '_mode', '_subsamples', '_samplen')
    _attrs = __slots__

    def __init__(self,
                 columnname: str,
                 samplen: str,
                 be_handles: BACKEND_ACCESSOR_MAP,
                 specs: Dict[KeyType, DataHashSpecsType],
                 mode: str,
                 *args, **kwargs):

        self._column_name = columnname
        self._samplen = samplen
        self._be_fs = be_handles
        self._subsamples = specs
        self._mode = mode
        self._stack: Optional[ExitStack] = None

    @property
    def _debug_(self):  # pragma: no cover
        return {
            '__class__': self.__class__,
            '_column_name': self._column_name,
            '_samplen': self._samplen,
            '_be_fs': self._be_fs,
            '_subsamples': self._subsamples,
            '_mode': self._mode,
            '_stack': self._stack._exit_callbacks if self._stack else self._stack,
        }

    def __repr__(self):
        res = f'{self.__class__}('\
              f'aset_name={self._column_name}, '\
              f'sample_name={self._samplen})'
        return res

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar {self.__class__.__name__} \
                \n    Arrayset Name            : {self._column_name}\
                \n    Sample Name              : {self._samplen}\
                \n    Mode (read/write)        : "{self._mode}"\
                \n    Number of Subsamples     : {self.__len__()}\n'
        p.text(res)

    def _ipython_key_completions_(self):
        """Let ipython know that any key based access can use the column keys

        Since we don't want to inherit from dict, nor mess with `__dir__` for
        the sanity of developers, this is the best way to ensure users can
        autocomplete keys.

        Returns
        -------
        list
            list of strings, each being one of the column keys for access.
        """
        return list(self.keys())

    def __enter__(self):
        self._enter_count += 1
        return self

    def __exit__(self, *exc):
        self._enter_count -= 1

    def _destruct(self):
        if isinstance(self._stack, ExitStack):
            self._stack.close()
        for attr in self._attrs:
            delattr(self, attr)

    def __getattr__(self, name):
        """Raise permission error after checkout is closed.

         Only runs after a call to :meth:`_destruct`, which is responsible for
         deleting all attributes from the object instance.
        """
        try:
            self.__getattribute__('_mode')  # once checkout is closed, this won't exist.
        except AttributeError:
            err = (f'Unable to operate on past checkout objects which have been '
                   f'closed. No operation occurred. Please use a new checkout.')
            raise PermissionError(err) from None
        return self.__getattribute__(name)

    @reader_checkout_only
    def __getstate__(self) -> dict:
        """ensure multiprocess operations can pickle relevant data.
        """
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __setstate__(self, state: dict) -> None:
        """ensure multiprocess operations can pickle relevant data.

        Technically should be decorated with @reader_checkout_only, but since
        at instance creation that is not an attribute, the decorator won't
        know. Since only readers can be pickled, This isn't much of an issue.
        """
        for slot, value in state.items():
            setattr(self, slot, value)

    def __len__(self) -> int:
        return len(self._subsamples)

    def __contains__(self, key: KeyType) -> bool:
        return key in self._subsamples

    def __iter__(self) -> Iterable[KeyType]:
        yield from self.keys()

    def __getitem__(self, key: GetKeysType) -> Union[np.ndarray, KeyArrMap]:
        """Retrieve data for some subsample key via dict style access conventions.

        .. seealso:: :meth:`get`

        Parameters
        ----------
        key : GetKeysType
            Sample key to retrieve from the column. Alternatively, ``slice``
            syntax can be used to retrieve a selection of subsample
            keys/values. An empty slice (``: == slice(None)``) or ``Ellipsis``
            (``...``) will return all subsample keys/values. Passing a
            non-empty slice (``[1:5] == slice(1, 5)``) will select keys to
            retrieve by enumerating all subsamples and retrieving the element
            (key) for each step across the range. Note: order of enumeration is
            not guaranteed; do not rely on any ordering observed when using
            this method.

        Returns
        -------
        Union[:class:`numpy.ndarray`, KeyArrMap]
            Sample array data corresponding to the provided key. or dictionary
            of subsample keys/data if Ellipsis or slice passed in as key.

        Raises
        ------
        KeyError
            if no sample with the requested key exists.
        ValueError
            If the keys argument if not a valid format.
        """
        # select subsample(s) with regular keys
        if isinstance(key, (str, int)):
            spec = self._subsamples[key]
            return self._be_fs[spec.backend].read_data(spec)
        # select all subsamples
        elif key is Ellipsis:
            res = {}
            for subsample, spec in self._subsamples.items():
                res[subsample] = self._be_fs[spec.backend].read_data(spec)
            return res
        # slice subsamples by sorted order of keys
        elif isinstance(key, slice):
            res = {}
            subsample_spec_slice = tuple(self._subsamples.items())[key]
            for subsample, spec in subsample_spec_slice:
                spec = self._subsamples[subsample]
                res[subsample] = self._be_fs[spec.backend].read_data(spec)
            return res
        else:
            raise TypeError(f'key {key} type {type(key)} not valid.')

    @property
    def _enter_count(self):
        return self._be_fs['enter_count']

    @_enter_count.setter
    def _enter_count(self, value):
        self._be_fs['enter_count'] = value

    @property
    def _is_conman(self):
        return bool(self._enter_count)

    @property
    def sample(self) -> KeyType:
        """Return name (key) of this sample.
        """
        return self._samplen

    @property
    def column(self) -> str:
        """Return name (key) of column this sample is contained in.
        """
        return self._column_name

    @property
    def data(self) -> KeyArrMap:
        """Return dict mapping every subsample key / data value stored in the sample.

        Returns
        -------
        KeyArrMap
            Dictionary mapping subsample name(s) (keys) to their stored values
            as :class:`numpy.ndarray` instances.
        """
        return self[...]

    def _mode_local_aware_key_looper(self, local: bool) -> Iterable[KeyType]:
        """Generate keys for iteration with dict update safety ensured.

        Parameters
        ----------
        local : bool
            True if keys should be returned which only exist on the local
            machine. False if remote sample keys should be excluded.

        Returns
        -------
        Iterable[KeyType]
            Sample keys conforming to the `local` argument spec.
        """
        if local:
            if self._mode == 'r':
                yield from valfilter(lambda x: x.islocal, self._subsamples).keys()
            else:
                yield from tuple(valfilter(lambda x: x.islocal, self._subsamples).keys())
        else:
            if self._mode == 'r':
                yield from self._subsamples.keys()
            else:
                yield from tuple(self._subsamples.keys())

    @property
    def contains_remote_references(self) -> bool:
        """Bool indicating all subsamples in sample column exist on local disk.

        The data associated with subsamples referencing some remote server will
        need to be downloaded (``fetched`` in the hangar vocabulary) before
        they can be read into memory.

        Returns
        -------
        bool
            False if at least one subsample in the column references data
            stored on some remote server. True if all sample data is available
            on the machine's local disk.
        """
        return not all(map(lambda x: x.islocal, self._subsamples.values()))

    @property
    def remote_reference_keys(self) -> Tuple[KeyType]:
        """Compute subsample names whose data is stored in a remote server reference.

        Returns
        -------
        Tuple[KeyType]
            list of subsample keys in the column whose data references indicate
            they are stored on a remote server.
        """
        return tuple(valfilterfalse(lambda x: x.islocal, self._subsamples).keys())

    def keys(self, local: bool = False) -> Iterable[KeyType]:
        """Generator yielding the name (key) of every subsample.

        Parameters
        ----------
        local : bool, optional
            If True, returned keys will only correspond to data which is
            available for reading on the local disk, by default False.

        Yields
        ------
        Iterable[KeyType]
            Keys of one subsample at a time inside the sample.
        """
        yield from self._mode_local_aware_key_looper(local)

    def values(self, local: bool = False) -> Iterable[np.ndarray]:
        """Generator yielding the tensor data for every subsample.

        Parameters
        ----------
        local : bool, optional
            If True, returned values will only correspond to data which is
            available for reading on the local disk. No attempt will be made to
            read data existing on a remote server, by default False.

        Yields
        ------
        Iterable[:class:`numpy.ndarray`]
            Values of one subsample at a time inside the sample.
        """
        for key in self._mode_local_aware_key_looper(local):
            yield self[key]

    def items(self, local: bool = False) -> Iterable[Tuple[KeyType, np.ndarray]]:
        """Generator yielding (name, tensor) tuple for every subsample.

        Parameters
        ----------
        local : bool, optional
            If True, returned keys/values will only correspond to data which is
            available for reading on the local disk, No attempt will be made to
            read data existing on a remote server, by default False.

        Yields
        ------
        Iterable[Tuple[KeyType, np.ndarray]]
            Name and stored value for every subsample inside the sample.
        """
        for key in self._mode_local_aware_key_looper(local):
            yield (key, self[key])

    def get(self, key: KeyType, default: Any = None) -> np.ndarray:
        """Retrieve the data associated with some subsample key

        Parameters
        ----------
        key : GetKeysType
            The name of the subsample(s) to retrieve. Passing a single
            subsample key will return the stored :class:`numpy.ndarray`
        default : Any
            if a `key` parameter is not found, then return this value instead.
            By default, None.

        Returns
        -------
        np.ndarray
            :class:`numpy.ndarray` array data stored under subsample key
            if key exists, else default value if not found.
        """
        try:
            return self[key]
        except KeyError:
            return default


# ---------------- writer methods only after this point -------------------


class FlatSubsampleWriter(FlatSubsampleReader):

    __slots__ = ('_schema', '_txnctx', '_path')
    _attrs = __slots__ + FlatSubsampleReader.__slots__

    def __init__(self,
                 schema,
                 repo_path: Path,
                 aset_ctx: Optional[AsetTxnType] = None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._path = repo_path
        self._schema = schema
        self._txnctx = aset_ctx

    def __enter__(self):
        with ExitStack() as stack:
            self._txnctx.open_write()
            stack.callback(self._txnctx.close_write)
            if self._enter_count == 0:
                for k in self._be_fs.keys():
                    if k in ('enter_count', 'schema_spec'):
                        continue
                    stack.enter_context(self._be_fs[k])
            self._enter_count += 1
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self._enter_count -= 1
        if self._enter_count == 0:
            self._stack = None

    def __set_arg_validate(self, key: KeyType, value: np.ndarray):
        if not is_suitable_user_key(key):
            raise ValueError(f'Sample name `{key}` is not suitable.')
        isCompat = self._schema.verify_data_compatible(value)
        if not isCompat.compatible:
            raise ValueError(isCompat.reason)

    def __perform_set(self, key: KeyType, value: np.ndarray) -> None:
        """Internal write method. Assumes all arguments validated and context is open

        Parameters
        ----------
        key : KeyType
            subsample key to store
        value : np.ndarray
            tensor data to store
        """
        # full_hash = array_hash_digest(value)
        full_hash = self._schema.data_hash_digest(value)
        hashKey = hash_data_db_key_from_raw_key(full_hash)

        # check if data record already exists with given key
        dataRecKey = nested_data_db_key_from_names(self._column_name, self._samplen, key)
        existingDataRecVal = self._txnctx.dataTxn.get(dataRecKey, default=False)
        if existingDataRecVal:
            # check if data record already with same key & hash value
            existingDataRec = data_record_digest_val_from_db_val(existingDataRecVal)
            if full_hash == existingDataRec.digest:
                return

        # write new data if data hash does not exist
        existingHashVal = self._txnctx.hashTxn.get(hashKey, default=False)
        if existingHashVal is False:
            backendCode = self._schema.backend
            hashVal = self._be_fs[backendCode].write_data(value)
            self._txnctx.hashTxn.put(hashKey, hashVal)
            self._txnctx.stageHashTxn.put(hashKey, hashVal)
            hash_spec = backend_decoder(hashVal)
        else:
            hash_spec = backend_decoder(existingHashVal)
            if hash_spec.backend not in self._be_fs:
                # when adding data which is already stored in the repository, the
                # backing store for the existing data location spec may not be the
                # same as the backend which the data piece would have been saved in here.
                #
                # As only the backends actually referenced by a columns samples are
                # initialized (accessible by the column), there is no guarantee that
                # an accessor exists for such a sample. In order to prevent internal
                # errors from occurring due to an uninitialized backend if a previously
                # existing data piece is "saved" here and subsequently read back from
                # the same writer checkout, we perform an existence check and backend
                # initialization, if appropriate.
                fh = open_file_handles(backends=(hash_spec.backend,),
                                       path=self._path,
                                       mode='a',
                                       schema=self._schema)
                self._be_fs[hash_spec.backend] = fh[hash_spec.backend]

        # add the record to the db
        dataRecVal = data_record_db_val_from_digest(full_hash)
        self._txnctx.dataTxn.put(dataRecKey, dataRecVal)
        self._subsamples[key] = hash_spec

    def __setitem__(self, key: KeyType, value: np.ndarray) -> None:
        """Store a piece of data as a subsample. Convenience method to :meth:`add`.

        .. seealso::

            :meth:`add` for the actual method called.

            :meth:`update` for an implementation analogous to python's built
            in :meth:`dict.update` method which accepts a dict or iterable of
            key/value pairs to add in the same operation.

        Parameters
        ----------
        key : KeyType
            Key (name) of the subsample to add to the column.
        value : :class:`numpy.ndarray`
            Tensor data to add as the sample.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            self.__set_arg_validate(key, value)
            self.__perform_set(key, value)

    def append(self, value: np.ndarray) -> KeyType:
        """Store some data in a subsample with an automatically generated key.

        This method should only be used if the context some piece of data is
        used in is independent from it's value (ie. when reading data back,
        there is no useful information which needs to be conveyed between the
        data source's name/id and the value of that piece of information.)
        Think carefully before going this route, as this posit does not apply
        to many common use cases.

        .. seealso::

            In order to store the data with a user defined key, use
            :meth:`update` or :meth:`__setitem__`

        Parameters
        ----------
        value: :class:`numpy.ndarray`
            Piece of data to store in the column.

        Returns
        -------
        KeyType
            Name of the generated key this data is stored with.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            key = generate_sample_name()
            self.__set_arg_validate(key, value)
            self.__perform_set(key, value)
            return key

    def update(self, other: Union[None, MapKeyArrType] = None, **kwargs) -> None:
        """Store data with the key/value pairs, overwriting existing keys.

        :meth:`update` implements functionality similar to python's builtin
        :meth:`dict.update` method, accepting either a dictionary or other
        iterable (of length two) listing key / value pairs.

        Parameters
        ----------
        other : Union[None, MapKeyArrType], optional
            Accepts either another dictionary object or an iterable of key/value
            pairs (as tuples or other iterables of length two). mapping sample
            names to :class:`np.ndarray` instances, If sample name is string type,
            can only contain alpha-numeric ascii characters (in addition to '-',
            '.', '_'). Int key must be >= 0. By default, None.
        **kwargs
            keyword arguments provided will be saved with keywords as subsample
            keys (string type only) and values as np.array instances.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            if other:
                if not isinstance(other, dict):
                    other = dict(other)
                else:
                    other = other.copy()
            elif other is None:
                other = {}
            if kwargs:
                # we have to merge kwargs dict with `other` before operating on
                # either so all validation and writing occur atomically
                other.update(kwargs)

            for key, val in other.items():
                self.__set_arg_validate(key, val)
            for key, val in other.items():
                self.__perform_set(key, val)

    def __delitem__(self, key: KeyType) -> None:
        """Remove a subsample from the column.`.

        .. seealso:: :meth:`pop` to simultaneously get value and delete.

        Parameters
        ----------
        key : KeyType
            Name of the sample to remove from the column.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            if key not in self._subsamples:
                raise KeyError(key)

            dbKey = nested_data_db_key_from_names(self._column_name, self._samplen, key)
            isRecordDeleted = self._txnctx.dataTxn.delete(dbKey)
            if isRecordDeleted is False:
                raise RuntimeError(
                    f'Internal error. Not able to delete key {key} from staging '
                    f'db even though existence passed in memory verification. '
                    f'Please report this message in full to the hangar development team.',
                    f'Specified key: <{type(key)} {key}>', f'Calculated dbKey: <{dbKey}>',
                    f'isRecordDeleted: <{isRecordDeleted}>', f'DEBUG STRING: {self._debug_}')
            del self._subsamples[key]

    def pop(self, key: KeyType) -> np.ndarray:
        """Retrieve some value for some key(s) and delete it in the same operation.

        Parameters
        ----------
        key : KeysType
            Sample key to remove

        Returns
        -------
        :class:`np.ndarray`
            Upon success, the value of the removed key.
        """
        value = self[key]
        del self[key]
        return value


class NestedSampleReader:

    __slots__ = ('_mode', '_column_name', '_samples',
                 '_be_fs', '_path', '_stack', '_schema')
    _attrs = __slots__

    def __init__(self,
                 columnname: str,
                 samples: Dict[KeyType, FlatSubsampleReader],
                 backend_handles: Dict[str, Any],
                 repo_path: Path,
                 mode: str,
                 schema=None,
                 *args, **kwargs):

        self._mode = mode
        self._column_name = columnname
        self._samples = samples
        self._be_fs = backend_handles
        self._path = repo_path
        self._stack: Optional[ExitStack] = None
        self._schema = schema

    def __repr__(self):
        res = f'{self.__class__}('\
              f'repo_pth={self._path}, '\
              f'columnname={self._column_name}, '\
              f'default_schema_hash={self._dflt_schema_hash}, '\
              f'isVar={self._schema_variable}, '\
              f'varMaxShape={self._schema_max_shape}, '\
              f'varDtypeNum={self._schema_dtype_num}, '\
              f'mode={self._mode})'
        return res

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar {self.__class__.__qualname__} \
                \n    Column Name              : {self._column_name}\
                \n    Access Mode              : {self._mode}\
                \n    Number of Samples        : {self.__len__()}\
                \n    Partial Remote Data Refs : {bool(self.contains_remote_references)}\
                \n    Contains Subsamples      : True\
                \n    Number of Subsamples     : {self.num_subsamples}\n'
        p.text(res)

    def _ipython_key_completions_(self):
        """Let ipython know that any key based access can use the column keys

        Since we don't want to inherit from dict, nor mess with `__dir__` for
        the sanity of developers, this is the best way to ensure users can
        autocomplete keys.

        Returns
        -------
        list
            list of strings, each being one of the column keys for access.
        """
        return list(self.keys())

    def __enter__(self):
        self._enter_count += 1
        return self

    def __exit__(self, *exc):
        self._enter_count -= 1

    def _destruct(self):
        if isinstance(self._stack, ExitStack):
            self._stack.close()
        self._close()
        for sample in self._samples.values():
            sample._destruct()
        for attr in self._attrs:
            delattr(self, attr)

    def __getattr__(self, name):
        """Raise permission error after checkout is closed.

         Only runs after a call to :meth:`_destruct`, which is responsible
         for deleting all attributes from the object instance.
        """
        try:
            self.__getattribute__('_mode')  # once checkout is closed, this won't exist.
        except AttributeError:
            err = (f'Unable to operate on past checkout objects which have been '
                   f'closed. No operation occurred. Please use a new checkout.')
            raise PermissionError(err) from None
        return self.__getattribute__(name)

    @reader_checkout_only
    def __getstate__(self) -> dict:
        """ensure multiprocess operations can pickle relevant data.
        """
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __setstate__(self, state: dict) -> None:
        """ensure multiprocess operations can pickle relevant data.

        Technically should be decorated with @reader_checkout_only, but since
        at instance creation the '_mode' is not a set attribute, the decorator won't
        know how to process. Since only readers can be pickled, This isn't much of an issue.
        """
        for slot, value in state.items():
            setattr(self, slot, value)

    def __getitem__(self, key: KeyType) -> FlatSubsampleReader:
        """Get the sample access class for some sample key.

        Parameters
        ----------
        key : KeyType
            Name of sample to retrieve

        Returns
        -------
        FlatSubsampleReader
            Sample accessor corresponding to the given key
        """
        return self._samples[key]

    def __iter__(self) -> Iterable[KeyType]:
        """Create iterator yielding an column sample keys.

        Yields
        -------
        Iterable[KeyType]
            Sample key contained in the column.
        """
        yield from self.keys()

    def __len__(self) -> int:
        """Find number of samples in the column
        """
        return len(self._samples)

    def __contains__(self, key: KeyType) -> bool:
        """Determine if some sample key exists in the column.
        """
        return key in self._samples

    def _open(self):
        for val in self._be_fs.values():
            try:
                # since we are storing non backend accessor information in the
                # be_fs weakref proxy for the purpose of memory savings, not
                # all elements have a `open` method
                val.open(mode=self._mode)
            except AttributeError:
                pass

    def _close(self):
        for val in self._be_fs.values():
            # since we are storing non backend accessor information in the
            # be_fs weakref proxy for the purpose of memory savings, not all
            # elements have a `close` method
            try:
                val.close()
            except AttributeError:
                pass

    @property
    def _enter_count(self):
        return self._be_fs['enter_count']

    @_enter_count.setter
    def _enter_count(self, value):
        self._be_fs['enter_count'] = value

    @property
    def _is_conman(self):
        return bool(self._enter_count)

    @property
    def column(self) -> str:
        """Name of the column.
        """
        return self._column_name

    @property
    def column_type(self):
        return self._schema.column_type

    @property
    def column_layout(self):
        return self._schema.column_layout

    @property
    def schema_type(self):
        return self._schema.schema_type

    @property
    def dtype(self):
        return self._schema.dtype

    @property
    def shape(self):
        try:
            return self._schema.shape
        except AttributeError:
            return None

    @property
    def iswriteable(self) -> bool:
        """Bool indicating if this column object is write-enabled.
        """
        return False if self._mode == 'r' else True

    def _mode_local_aware_key_looper(self, local: bool) -> Iterable[KeyType]:
        """Generate keys for iteration with dict update safety ensured.

        Parameters
        ----------
        local : bool
            True if keys should be returned which only exist on the local
            machine. False if remote sample keys should be excluded.

        Returns
        -------
        Iterable[KeyType]
            Sample keys conforming to the `local` argument spec.
        """
        if local:
            if self._mode == 'r':
                yield from valfilterfalse(lambda x: x.contains_remote_references, self._samples).keys()
            else:
                yield from tuple(valfilterfalse(lambda x: x.contains_remote_references, self._samples).keys())
        else:
            if self._mode == 'r':
                yield from self._samples.keys()
            else:
                yield from tuple(self._samples.keys())

    @property
    def contains_remote_references(self) -> bool:
        """Bool indicating all subsamples in sample column exist on local disk.

        The data associated with subsamples referencing some remote server will
        need to be downloaded (``fetched`` in the hangar vocabulary) before
        they can be read into memory.

        Returns
        -------
        bool
            False if at least one subsample in the column references data
            stored on some remote server. True if all sample data is available
            on the machine's local disk.
        """
        return all(map(lambda x: x.contains_remote_references, self._samples.values()))

    @property
    def remote_reference_keys(self) -> Tuple[KeyType]:
        """Compute subsample names whose data is stored in a remote server reference.

        Returns
        -------
        Tuple[KeyType]
            list of subsample keys in the column whose data references indicate
            they are stored on a remote server.
        """
        return tuple(valfilter(lambda x: x.contains_remote_references, self._samples).keys())

    @property
    def backend(self) -> str:
        """The default backend for the column.

        Returns
        -------
        str
            numeric format code of the default backend.
        """
        return self._schema.backend

    @property
    def backend_options(self) -> dict:
        """The config settings applied to the default storage backend filters.
        """
        return self._schema.backend_options

    @property
    def contains_subsamples(self) -> bool:
        """Bool indicating if sub-samples are contained in this column container.
        """
        return True

    @property
    def num_subsamples(self) -> int:
        """Calculate total number of subsamples existing in all samples in column
        """
        total = 0
        for sample in self._samples.values():
            total += len(sample)
        return total

    def keys(self, local: bool = False) -> Iterable[KeyType]:
        """Generator yielding the name (key) of every subsample.

        Parameters
        ----------
        local : bool, optional
            If True, returned keys will only correspond to data which is
            available for reading on the local disk, by default False.

        Yields
        ------
        Iterable[KeyType]
            Keys of one subsample at a time inside the sample.
        """
        yield from self._mode_local_aware_key_looper(local)

    def values(self, local: bool = False) -> Iterable[np.ndarray]:
        """Generator yielding the tensor data for every subsample.

        Parameters
        ----------
        local : bool, optional
            If True, returned values will only correspond to data which is
            available for reading on the local disk. No attempt will be made to
            read data existing on a remote server, by default False.

        Yields
        ------
        Iterable[:class:`numpy.ndarray`]
            Values of one subsample at a time inside the sample.
        """
        for key in self._mode_local_aware_key_looper(local):
            yield self[key]

    def items(self, local: bool = False) -> Iterable[Tuple[KeyType, np.ndarray]]:
        """Generator yielding (name, tensor) tuple for every subsample.

        Parameters
        ----------
        local : bool, optional
            If True, returned keys/values will only correspond to data which is
            available for reading on the local disk, No attempt will be made to
            read data existing on a remote server, by default False.

        Yields
        ------
        Iterable[Tuple[KeyType, np.ndarray]]
            Name and stored value for every subsample inside the sample.
        """
        for key in self._mode_local_aware_key_looper(local):
            yield (key, self[key])

    def get(self, key: GetKeysType, default: Any = None) -> FlatSubsampleReader:
        """Retrieve tensor data for some sample key(s) in the column.

        Parameters
        ----------
        key : GetKeysType
            The name of the subsample(s) to retrieve
        default : Any
            if a `key` parameter is not found, then return this value instead.
            By default, None.

        Returns
        -------
        FlatSubsampleReader:
            Sample accessor class given by name ``key`` which can be used to
            access subsample data.
        """
        try:
            return self[key]
        except KeyError:
            return default


# ---------------- writer methods only after this point -------------------


class NestedSampleWriter(NestedSampleReader):

    __slots__ = ('_txnctx',)
    _attrs = __slots__ + NestedSampleReader.__slots__

    def __init__(self,
                 aset_ctx: Optional[AsetTxnType] = None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._txnctx = aset_ctx

    def __enter__(self):
        with ExitStack() as stack:
            self._txnctx.open_write()
            stack.callback(self._txnctx.close_write)
            if self._enter_count == 0:
                for k in tuple(self._be_fs.keys()):
                    if k in ('enter_count', 'schema_spec'):
                        continue
                    stack.enter_context(self._be_fs[k])
            self._enter_count += 1
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self._enter_count -= 1

    def __set_arg_validate(self, sample_key: KeyType, subsample_map: MapKeyArrType):

        if not is_suitable_user_key(sample_key):
            raise ValueError(f'Sample name `{sample_key}` is not suitable.')

        for subsample_key, subsample_val in subsample_map.items():
            if not is_suitable_user_key(subsample_key):
                raise ValueError(f'Sample name `{sample_key}` is not suitable.')
            isCompat = self._schema.verify_data_compatible(subsample_val)
            if not isCompat.compatible:
                raise ValueError(isCompat.reason)

    def __perform_set(self, key: KeyType, value: MapKeyArrType) -> None:
        if key in self._samples:
            self._samples[key].update(value)
        else:
            self._samples[key] = FlatSubsampleWriter(
                schema=proxy(self._schema),
                aset_ctx=proxy(self._txnctx),
                repo_path=self._path,
                columnname=self._column_name,
                samplen=key,
                be_handles=proxy(self._be_fs),
                specs={},
                mode='a')
            try:
                self._samples[key].update(value)
            except Exception as e:
                del self._samples[key]
                raise e

    def __setitem__(self, key: KeyType, value: MapKeyArrType) -> None:
        """Store some subsample key / subsample data map, overwriting existing keys.

        .. seealso::

            :meth:`add` for the actual implementation of the method and docstring
            for this methods parameters
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            value = dict(value)
            self.__set_arg_validate(key, value)
            self.__perform_set(key, value)

    def update(self,
               other: Union[None, Dict[KeyType, MapKeyArrType],
                            Sequence[Sequence[Union[KeyType, MapKeyArrType]]]] = None,
               **kwargs) -> None:
        """Store some data with the key/value pairs, overwriting existing keys.

        :meth:`update` implements functionality similar to python's builtin
        :py:`dict.update` method, accepting either a dictionary or other
        iterable (of length two) listing key / value pairs.

        Parameters
        ----------
        other : Union[None, Dict[KeyType, MapKeyArrType],
                      Sequence[Sequence[Union[KeyType, MapKeyArrType]]]]

            Dictionary mapping sample names to subsample data maps. Or Sequence
            (list or tuple) where element one is the sample name and element
            two is a subsample data map.

        **kwargs :
            keyword arguments provided will be saved with keywords as sample
            keys (string type only) and values as a mapping of subarray keys
            to :class:`np.ndarray` instances.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            if other:
                if not isinstance(other, dict):
                    other = dict(other)
                else:
                    other = other.copy()
            elif other is None:
                other = {}
            if kwargs:
                # we merge kwargs dict with `other` before operating on either
                # so all necessary validation and writing occur atomically
                other.update(kwargs)

            for sample in tuple(other.keys()):
                other[sample] = dict(other[sample])

            for key, val in other.items():
                self.__set_arg_validate(key, val)
            for key, val in other.items():
                self.__perform_set(key, val)

    def __delitem__(self, key: KeyType) -> None:
        """Remove a sample (including all contained subsamples) from the column.

        .. seealso::

            :meth:`delete` for the actual implementation of the method and
            docstring for this methods parameters
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            sample = self._samples[key]
            subsample_keys = list(sample.keys())
            for subkey in subsample_keys:
                del sample[subkey]

            self._samples[key]._destruct()
            del self._samples[key]

    def pop(self, key: KeyType) -> Dict[KeyType, KeyArrMap]:
        """Retrieve some value for some key(s) and delete it in the same operation.

        Parameters
        ----------
        key : KeysType
            sample key to remove

        Returns
        -------
        Dict[KeyType, KeyArrMap]
            Upon success, a nested dictionary mapping sample names to a dict of
            subsample names and subsample values for every sample key passed
            into this method.
        """
        res = self._samples[key].data
        del self[key]
        return res

    def change_backend(self, backend: str, backend_options: Optional[dict] = None):
        """Change the default backend and filters applied to future data writes.

        .. warning::

           This method is meant for advanced users only. Please refer to the
           hangar backend codebase for information on accepted parameters and
           options.

        Parameters
        ----------
        backend : str
            Backend format code to swtich to.
        backend_options
            Backend option specification to use (if specified). If left to default
            value of None, then default options for backend are automatically used.

        Raises
        ------
        RuntimeError
            If this method was called while this column is invoked in a
            context manager
        ValueError
            If the backend format code is not valid.
        """
        if self._is_conman:
            raise RuntimeError('Cannot call method inside column context manager.')

        self._schema.change_backend(backend, backend_options=backend_options)

        schema_digest = self._schema.schema_hash_digest()
        columnSchemaKey = schema_db_key_from_column(self._column_name, layout=self.column_layout)
        columnSchemaVal = schema_db_val_from_spec(self._schema.schema)
        hashSchemaKey = schema_hash_db_key_from_digest(schema_digest)

        # -------- set vals in lmdb only after schema is sure to exist --------

        with self._txnctx.write() as ctx:
            ctx.dataTxn.put(columnSchemaKey, columnSchemaVal)
            ctx.hashTxn.put(hashSchemaKey, columnSchemaVal, overwrite=False)

        if self._schema.backend not in self._be_fs:
            fhands = open_file_handles(
                backends=[self._schema.backend],
                path=self._path,
                mode='a',
                schema=self._schema)
            self._be_fs[self._schema.backend] = fhands[self._schema.backend]
        else:
            self._be_fs[self._schema.backend].close()
        self._be_fs[self._schema.backend].open(mode='a')
        self._be_fs[self._schema.backend].backend_opts = self._schema.backend_options
        return