"""A test suite running on sample files of formats that we (attempt to) support.

This should help find breaking changes on our end or in upstream dependencies.

The different test cases are based on sample files that were actually recorded by various software programs.

They are stored in `./test_files`.
"""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from spexread import read_spe_file

from OES_toolbox.file_handling import FileLoader

if TYPE_CHECKING:
    from xarray import DataArray

from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_index_equal
from xarray.testing import assert_identical


def hash_array_like(df:pd.DataFrame|np.ndarray):
    """Hash an array-like object.
    
    If the input is a pandas DataFrame, add dtypes information to the hash as well.
    Dataframes will convert to numpy first.
    """
    hashed = hashlib.sha256(str(df.shape).encode())
    if isinstance(df,pd.DataFrame):
        hashed.update(df.dtypes.to_numpy().tobytes())
        df = df.to_numpy()
    hashed.update(df.tobytes())
    return hashed.hexdigest()



class TestSupportedFiles:
        """Class containing test cases to test supported file types with concrete example files.
        
        Tests files that we handle ourselves, and those for which we have a dependency.

        This enables us to detect if file support breaks even if changes happen upstream.

        For each example file, specific information (e.g. shape, sum, etc.) is hardcoded in the tests to validate consistency.
        """


        def test_read_horiba_txt(self,horiba_file):
            """Test the file provided as a sample for issue #4."""
            wl,data,_ = FileLoader.read_horiba_txt("./tests/test_files/Horiba.txt")
            assert wl.shape == (2048,)
            assert data.shape == (2048,300)
            assert data.shape[0] == wl.shape[0]
            assert hash_array_like(wl) == "55be8ffed4f4d9f49ad9a4ee0e10a56c316cc63fac615d3d31a3862ab153d4fd"
            assert hash_array_like(data) == "d4fb8e612f8962b3a04483bfcb3c6e3bb3b7c4d7d09f096192a0a03fbe741b59"
            assert_index_equal(data.index,pd.RangeIndex(2048))
            assert_allclose([wl.min(),wl.mean(),wl.max()],[190.037,540.8279,884.273])
            assert_allclose([data.mean().min(),data.mean().mean(),data.mean().max()],[4491.5849,4703.6766,66363.9409])
            assert_frame_equal(data,horiba_file[1])
            assert_allclose(wl,horiba_file[0])

        def test_read_andor_sif_numpy(self,Andor_kinetic_sif_file:"DataArray"):
            """Test a sample Andor SIF file containing a kinetic series, using `sif_parser` in `numpy` mode.

            For `sif_parser` version 0.3.5 and earlier this relies on our custom extraction the wavelength calibration; this will not work for step-and-glue.
            """
            wl,data = FileLoader.read_andor_sif(Path("./tests/test_files/Andor_kinetic.sif"))
            assert data.shape == (100,1,1024)
            assert hash_array_like(wl) == "200810d5daf15cb3b6d4bdf2e685f4deb0ed9c03e78b86bd3794430676351bf8"
            assert hash_array_like(data) == "a72d35208d145080e9901814942ed3e8df559a6ca1af68e43ac9330c556c5bb9"
            assert_allclose([wl.min(), wl.mean(),wl.max()],[452.4057,487.0280,521.4877])
            assert_allclose([data.min(),data.mean(),data.max()],[0,19029.64,186700],atol=0.004) # on py 3.10 an 3.11 mean will be 19029.637
            assert_allclose(data,Andor_kinetic_sif_file.data)

        def test_read_andor_sif_xarray(self,Andor_kinetic_sif_file:"DataArray"):
            """Test a sample Andor SIF file containing a kinetic series, using `sif_parser` in `xarray` mode.

            For `sif_parser` version 0.3.5 and earlier this relies on our custom extraction the wavelength calibration; this will not work for step-and-glue.
            """
            data =FileLoader.read_andor_sif_xarray(Path("./tests/test_files/Andor_kinetic.sif"))
            assert data.shape == (100,1,1024)
            assert data.size == 102400
            assert data.attrs['size'] == (1024,1)
            assert data.shape == (100,1,1024)
            assert data.dims == ('Time', 'height', 'width')
            assert data.attrs['SifVersion'] == 65567
            assert data.calibration.dims == ('width',)
            # The hash of the data content should match the previous test `test_read_andor_sif_numpy` 
            assert hash_array_like(data.calibration.data) == "200810d5daf15cb3b6d4bdf2e685f4deb0ed9c03e78b86bd3794430676351bf8"
            assert hash_array_like(data.data) == "a72d35208d145080e9901814942ed3e8df559a6ca1af68e43ac9330c556c5bb9"
            assert_allclose([data.calibration.min(),data.calibration.mean(),data.calibration.max()],[452.4057,487.0280,521.4877])
            assert_allclose([data.min(),data.mean(),data.max()],[0,19029.64,186700],rtol=2e-7) # on py 3.10 an 3.11 mean() will be 19029.637, meaning sub 2e-7 relative difference
            assert_allclose(data.Time,np.zeros(100))
            assert_identical(data,Andor_kinetic_sif_file)

        def test_read_avantes_raw8(self,Avantes_raw8_demo_file):
            """Read a file created with AvaSoft 8 with a 'virtual demo spectrometer' (e.g. with no physical device attached.)"""
            data = FileLoader.read_avantes_binary(Path("./tests/test_files/Avasoft8_demo.raw8"))
            assert len(data)==1 # contains 1 channel
            assert data[0].ID.SerialNumber == '00000000'
            assert data[0].data.shape == (1615,4) # 4 vectors, wavelength, signal,dark,ref
            assert hash_array_like(data[0].data) == "ff6690dc23646cb08eab73d527546114fff2cb9fb44bf65a9269327b42780bec"
            assert data[0]==data.channels[0]
            assert round(data[0].exposure,8)==1.04999995
            assert data[0].wavelength.min()==174.029
            assert data[0].wavelength.max()==1100.3231
            assert_allclose(data[0].data,Avantes_raw8_demo_file)

        def test_read_ThorSpectra_txt(self,ThorSpectra_demo_file):
            """Read a text file exported from ThorSpectra using a demo spectrometer."""
            data = FileLoader._read_generic_text(Path("./tests/test_files/ThorSpectra_demo.txt"))
            assert data.shape == (2048,2)
            assert_allclose(data.mean(),[587.0918,151.64063])
            assert hash_array_like(ThorSpectra_demo_file) == "295fddde352b493e00ed521b85483146b402cf5c812ac5eb33ceb03e485e9041"
            assert_frame_equal(data,ThorSpectra_demo_file)