	
	
	function Phi_full = build_bspline_basis( x, full_knots, degree )
		% x          : N × 1 input values
		% full_knots : full (clamped) knot vector with boundary knots repeated p+1 times
		% degree     : spline degree p (e.g. p = 3 for cubic)
		% Phi        : N × nBasis design matrix, Phi(i,j) = B_j(x_i)

		% Construct full knot vector
		N = length( x );

		% Number of basis functions
		nBasis = length( full_knots ) - degree - 1;
		Phi = zeros( N, nBasis );
        
		for j = 1 : 1 : nBasis
			coeffs = zeros( 1, nBasis ); 
			coeffs( 1, j ) = 1;
			sp = spmak( full_knots, coeffs );
			Phi( :, j ) = fnval( sp, x );
		end
		
		Phi_full = Phi;
		
	end