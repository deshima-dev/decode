# coding: utf-8

# public items
__all__ = [
    'MPPool'
]

# standard library
import multiprocessing as mp

# dependent packages
import numpy as np


# classes
class MPPool(object):
    def __init__(self, n_processes=None):
        """Initialize a process pool object.

        Args:
            n_processes (int): The number of processes to be created. Default is
                <CPU count of your machine> -1 (one thread is saved for backup).

        """
        self.params = {
            'n_processes': n_processes or mp.cpu_count()-1,
        }

    def map(self, func, *sequences):
        """Return a list of the results of applying the function to the sequence.

        If self.mpcompatible is True, mapping is multiprocessed with the spacified
        number of processes (default is <CPU count> - 1). If False, mapping is
        singleprocessed (equivalent to the bulitin map function).

        Args:
            func (function): Applying function.
            sequences (lists): Lists of items to which function is applied.

        Returns:
            results (list): The results of applying the function to the sequence.

        """
        if self.mpcompatible:
            return self._mpmap(func, *sequences)
        else:
            return list(map(func, *sequences))

    @property
    def mpcompatible(self):
        """Whether your NumPy/SciPy is compatible with multiprocessing."""
        lapack_opt_info = np.__config__.lapack_opt_info

        if not 'libraries' in lapack_opt_info:
            return False
        else:
            libs = lapack_opt_info['libraries']
            mkl = any([('mkl' in lib) for lib in libs])
            blas = any([('blas' in lib) for lib in libs])
            atlas = any([('atlas' in lib) for lib in libs])
            return any([mkl, blas, atlas])

    def _mpmap(self, func, *sequences):
        """Multiprosessing map function that can work with non-local function."""
        n_processes = np.min([len(sequences[0]), self.n_processes])
        mpsequences = [[] for i in range(n_processes)]
        procs, parents, results = [], [], []

        def pfunc(child, psequences):
            child.send(list(map(func, *psequences)))
            child.close()

        for sequence in sequences:
            idxs = np.array_split(range(len(sequence)), n_processes)
            sseqs = [sequence[idx.min():idx.max()+1] for idx in idxs]
            for n in range(len(sseqs)):
                mpsequences[n].append(sseqs[n])

        for psequences in mpsequences:
            parent, child = mp.Pipe()
            proc = mp.Process(target=pfunc, args=(child, psequences))
            procs.append(proc)
            parents.append(parent)

        for proc in procs:
            proc.start()

        for parent in parents:
            results += parent.recv()

        for proc in procs:
            proc.join()

        return results

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return 'MPPool({0})'.format(self.params)
