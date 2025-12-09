	
	
function Phi_full = build_bspline_basis( x, full_knots, degree )
		% x: N × 1 input values
		% knots: vector of interior knots (no boundary knots)
		% degree: degree of B-spline (3 for cubic)

		% Construct full knot vector
		N = length( x );

		% Number of basis functions
		nBasis = length( full_knots ) - degree - 1;
		Phi = zeros( N, nBasis );
		
		% disp( 'Full knot sequence:' );
		% disp( full_knots );
        
		for j = 1 : 1 : nBasis
			coeffs = zeros( 1, nBasis ); 
			coeffs( 1, j ) = 1;
			sp = spmak( full_knots, coeffs );
			Phi( :, j ) = fnval( sp, x );
		end
		
		Phi_full = Phi;
	end