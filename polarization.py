import solpolpy as spp

def resolve_bpb(file_list):
    """
    Resolve B and pB from a list of input files.

    Parameters
    ----------
    file_list : list of str
        List of file paths (e.g., stereo_0.fts, stereo_120.fts, stereo_240.fts)

    Returns
    -------
    tB : numpy.ndarray
        Total brightness (B)
    pB : numpy.ndarray
        Polarized brightness (pB)
    outbpb : NDCollection
        Full output from spp.resolve (returned for flexibility)
    """
    outbpb = spp.resolve(file_list, 'BpB')

    tB = outbpb['B'].data
    pB = outbpb['pB'].data

    return tB, pB, outbpb