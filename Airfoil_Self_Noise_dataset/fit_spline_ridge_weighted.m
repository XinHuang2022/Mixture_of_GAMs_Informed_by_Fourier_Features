	
	
	function [ theta, f_j, P ] = fit_spline_ridge_weighted( Phi, r, lambda, w_vec )
		% Fit ridge regression with penalty: ||r - Phi * theta||^2 + lambda * ||D2 * theta||^2

		% Build penalty matrix approximating second derivatives
		D = diff( eye( size( Phi, 2 ) ), 2 );   % second-difference matrix
		P = D' * D;                      % roughness penalty
		
		w_vec = w_vec( : );
		
		A_mat_temp = Phi' * ( w_vec .* Phi ) + lambda * P;
		b_vec_temp = Phi' * ( w_vec .* r );

        warning('off', 'MATLAB:nearlySingularMatrix');
        warning('off', 'MATLAB:singularMatrix');
		theta = A_mat_temp \ b_vec_temp;
        warning('on', 'MATLAB:nearlySingularMatrix');
        warning('on', 'MATLAB:singularMatrix');

		% theta = ( Phi' * Phi + lambda * P ) \ ( Phi' * r );
		f_j = Phi * theta;
	end