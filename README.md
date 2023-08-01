# membrane_fluctuations
scripts for grid based fourier analysis of simulated membranes ( see also mferguder )

fourier.py: 

            INPUTS - Arrays of (head, tail, and surface) positions and box sizes

            OUTPUTS - Arrays of Fourier coefficients modulus squared (and sigmas) for shape fluctuations and
            (anti-symmetrized and symmetrized, parallel and transverse) director fluctuations + q vectors

analysis.py:

	INPUTS - Arrays of Fourier coefficients modulus squared (and sigmas) for shape fluctuations and
            (anti-symmetrized and symmetrized, parallel and transverse) director fluctuations + q vectors

 	OUTPUTS - Parameter estimates for fit to theory (See assorted works of M.Terzi,M.F.Erguder,M.Deserno,F.Brown),
  	Spectra plots with fit, etc.
