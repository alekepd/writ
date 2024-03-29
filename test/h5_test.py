"""Test h5 readers.

Note that some SchemaH5 does not have a stable order of output between python sessions; 
as a result, we do not rely on chunk ordering being stable.
"""

from typing import Final
import pytest
from tools import arrhash, ihash, ghash

from writ.read import SchemaH5


@pytest.mark.h5py
def test_schemah5_read(
    aaqaa_coord_h5: SchemaH5, aaqaa_coord_force_h5: SchemaH5
) -> None:
    """Test accuracy of basic single-array SchemaH5 reading.

    Content hash is checked against saved values and internal consistency.
    """
    CORRECT_COORD_HASH: Final = 17177354416086495880
    CORRECT_COORD_FORCE_HASH: Final = 13910756345033442274
    # the read entries are size one tuples, so we extract the array via [0]
    coord_content_hash = ihash(aaqaa_coord_h5, hasher=ghash, order=True)
    assert CORRECT_COORD_HASH == coord_content_hash
    # the read entries are size one tuples, so we extract the array via [0]
    coord_force_content_hash = ihash(aaqaa_coord_force_h5, hasher=ghash, order=True)
    assert CORRECT_COORD_FORCE_HASH == coord_force_content_hash
    assert coord_content_hash != coord_force_content_hash


@pytest.mark.h5py
def test_schemah5_strided_read(
    aaqaa_coord_h5: SchemaH5, aaqaa_coord_h5_stride_2: SchemaH5
) -> None:
    """Test accuracy of strided SchemaH5 reading.

    Data is read using built in striding and compared to manual striding.
    """
    # the read entries are size one tuples, so we extract the array via [0]
    strided_content_hash = ihash(
        (x[0] for x in aaqaa_coord_h5_stride_2), hasher=arrhash, order=True
    )
    manual_strided_content_hash = ihash(
        (x[0][::2] for x in aaqaa_coord_h5), hasher=arrhash, order=True
    )
    assert strided_content_hash == manual_strided_content_hash


@pytest.mark.h5py
def test_schemah5_ids(aaqaa_coord_h5_with_ids: SchemaH5) -> None:
    """Test accuracy id_included SchemaH5 reading.

    Hash is compared to saved value.
    """
    CORRECT_HASH: Final = 16610446184277274991
    content_hash = ihash((x for x in aaqaa_coord_h5_with_ids), hasher=ghash, order=True)
    assert CORRECT_HASH == content_hash


@pytest.mark.h5py
def test_schemah5_singleton(
    aaqaa_coord_h5_singleton: SchemaH5, aaqaa_coord_h5: SchemaH5
) -> None:
    """Test accuracy of singleton read.

    Hash is compared to saved value and output of reading without id.
    """
    CORRECT_HASH: Final = 5128426097335556466
    # the read entries are size one tuples, so we extract the array via [0]
    content_hash = ihash(
        (x for x in aaqaa_coord_h5_singleton), hasher=arrhash, order=True  # type: ignore
    )
    assert CORRECT_HASH == content_hash
    nos_content_hash = ihash((x[0] for x in aaqaa_coord_h5), hasher=arrhash, order=True)
    assert nos_content_hash == content_hash
