	
	
	function [ theta, f_j, P ] = fit_spline_ridge( Phi, r, lambda )
		% Fit ridge regression with penalty: ||r - Phi * theta||^2 + lambda * ||D2 * theta||^2

		% Build penalty matrix approximating second derivatives
		D = diff( eye( size( Phi, 2 ) ), 2 );   % second-difference matrix
		P = D' * D;                      % roughness penalty

		theta = ( Phi' * Phi + lambda * P ) \ ( Phi' * r );
		f_j = Phi * theta;
	end