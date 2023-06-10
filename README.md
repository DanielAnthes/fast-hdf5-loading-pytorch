# Fast HDF5 File Loading for PyTorch

This library implements a simple dataloader using datapipes for chunked streaming of hdf5 files.
This minimizes I/O calls to the file system and thus greatly speeds up data loading.
